"""Apply the bounded semantic-subgoal interface to the pinned SysNav source.

This does not replace TARE: its collision-filtered candidate graph, coverage
planner, graph shortest paths and waypoint publisher remain responsible for
geometry. Save the resulting git diff in patches/ when changing this adapter.
"""
from pathlib import Path
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("source", type=Path)
args = ap.parse_args()
p = args.source / "src/exploration_planner/tare_planner"
header = p / "include/sensor_coverage_planner/sensor_coverage_planner_ground.h"
source = p / "src/sensor_coverage_planner/sensor_coverage_planner_ground.cpp"
h, s = header.read_text(), source.read_text()
if "mission_subgoal_sub_" in h:
    raise SystemExit("Already patched; refusing to apply twice")
h = h.replace("  void execute();", """  void execute();
  void PublishMissionCandidates();
  rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr mission_subgoal_sub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr mission_candidates_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr mission_path_pub_;
  Eigen::Vector3d mission_goal_ = Eigen::Vector3d::Zero();
  double mission_goal_deadline_ = -1.0;
  int mission_revision_ = 0;""")
needle = "  execution_timer_ = this->create_wall_timer("
assert needle in s
s = s.replace(needle, """  mission_candidates_pub_ = this->create_publisher<std_msgs::msg::String>("/mission/candidates", 1);
  mission_path_pub_ = this->create_publisher<nav_msgs::msg::Path>("/mission/path", 1);
  mission_subgoal_sub_ = this->create_subscription<geometry_msgs::msg::PointStamped>(
      "/mission/subgoal", 1, [this](geometry_msgs::msg::PointStamped::ConstSharedPtr msg) {
        if (msg->header.frame_id != "map") return;
        double age = this->now().seconds() - rclcpp::Time(msg->header.stamp).seconds();
        if (age < -0.1 || age > 2.0) return;
        Eigen::Vector3d target(msg->point.x, msg->point.y, msg->point.z);
        if (!target.allFinite() || !viewpoint_manager_->InLocalPlanningHorizon(target)) return;
        int candidate = viewpoint_manager_->GetNearestCandidateViewPointInd(target);
        if (candidate < 0) return;
        auto point = viewpoint_manager_->GetViewPointPosition(candidate);
        if ((target - Eigen::Vector3d(point.x, point.y, point.z)).norm() > 0.4) return;
        mission_goal_ = target;
        mission_goal_deadline_ = this->now().seconds() + 25.0;
      });

""" + needle)
needle = "void SensorCoveragePlanner3D::PublishWaypoint() {"
assert needle in s
s = s.replace(needle, """void SensorCoveragePlanner3D::PublishMissionCandidates() {
  json result;
  result["revision"] = ++mission_revision_;
  result["time"] = this->now().seconds();
  result["candidates"] = json::array();
  Eigen::Vector3d start(robot_position_.x, robot_position_.y, robot_position_.z);
  std::vector<int> bins(12, -1);
  std::vector<double> scores(12, 1e9);
  for (int ind : viewpoint_manager_->GetViewPointCandidateIndices()) {
    auto pos = viewpoint_manager_->GetViewPointPosition(ind);
    Eigen::Vector3d delta(pos.x-start.x(), pos.y-start.y(), 0);
    double dist = delta.norm();
    if (dist < 1.0 || dist > 4.5) continue;
    int bin = std::min(11, std::max(0, int((atan2(delta.y(), delta.x()) + M_PI) / (2*M_PI) * 12)));
    double score = std::abs(dist-2.8);
    if (score < scores[bin]) { scores[bin] = score; bins[bin] = ind; }
  }
  for (int i=0; i<12; ++i) {
    if (bins[i] < 0) continue;
    auto pos = viewpoint_manager_->GetViewPointPosition(bins[i]);
    Eigen::Vector3d goal(pos.x, pos.y, pos.z);
    auto path = viewpoint_manager_->GetViewPointShortestPath(start, goal);
    if (path.poses.size() < 2) continue;
    result["candidates"].push_back({{"id", i}, {"xyz", {pos.x, pos.y, pos.z}},
                                    {"path_nodes", path.poses.size()}});
  }
  std_msgs::msg::String msg;
  msg.data = result.dump();
  mission_candidates_pub_->publish(msg);
}

void SensorCoveragePlanner3D::PublishWaypoint() {
  PublishMissionCandidates();
  if (this->now().seconds() < mission_goal_deadline_) {
    Eigen::Vector3d start(robot_position_.x, robot_position_.y, robot_position_.z);
    auto path = viewpoint_manager_->GetViewPointShortestPath(start, mission_goal_);
    geometry_msgs::msg::PointStamped target;
    target.point.x = start.x(); target.point.y = start.y(); target.point.z = start.z();
    if ((mission_goal_ - start).head<2>().norm() < 0.5) {
      mission_goal_deadline_ = -1.0;
    } else if (path.poses.size() >= 2) {
      // Follow the graph one short segment at a time, without extending across walls.
      target.point = path.poses.back().pose.position;
      for (auto &pose : path.poses) {
        auto &p = pose.pose.position;
        if (std::hypot(p.x-start.x(), p.y-start.y()) > 0.8) { target.point = p; break; }
      }
    } else {
      mission_goal_deadline_ = -1.0;  // Lost reachability: emit hold for this cycle.
    }
    path.header.frame_id = kWorldFrameID; path.header.stamp = this->now();
    mission_path_pub_->publish(path);
    misc_utils_ns::Publish(shared_from_this(), waypoint_pub_, target, kWorldFrameID);
    return;
  }
""")
s = s.replace("void SensorCoveragePlanner3D::execute() {", "void SensorCoveragePlanner3D::execute() {\n  if (this->now().seconds() <= 0.0 || odomTime <= 0.0) return;  // Wait for simulator clock and odometry.")
s = s.replace("  double lx = 12.0;", "  double lx = 0.0;  // Hold until observed geometry produces a valid path.")
header.write_text(h)
source.write_text(s)
manager = p / "src/viewpoint_manager/viewpoint_manager.cpp"
v = manager.read_text()
v = v.replace("void ViewPointManager::CheckViewPointRoomBoundaryCollision()\n{", """void ViewPointManager::CheckViewPointRoomBoundaryCollision()
{
  // Room-free semantic task mode: use geometric reachability until an online
  // room mask is available. This does not manufacture room identities.
  if (room_mask_.empty()) {
    for (int i=0; i<vp_.kViewPointNumber; ++i) SetViewPointInRoom(i, true, true);
    return;
  }""")
manager.write_text(v.replace("    return;\n  } \n", "    return;\n  }\n"))
grid = p / "src/grid_world/grid_world.cpp"
g = grid.read_text()
g = g.replace("namespace grid_world_ns\n{", """namespace grid_world_ns
{
static int SafeRoomId(const cv::Mat& mask, const Eigen::Vector3i& voxel) {
  if (mask.empty() || voxel.x()<0 || voxel.y()<0 || voxel.x()>=mask.rows || voxel.y()>=mask.cols) return -1;
  return mask.at<int>(voxel.x(), voxel.y());
}
""")
for var in ("viewpoint_pos_voxel", "connection_point_voxel"):
    g = g.replace(f"room_mask_.at<int>({var}.x(), {var}.y())", f"SafeRoomId(room_mask_, {var})")
grid.write_text(g)
local = p / "src/local_coverage_planner/local_coverage_planner.cpp"
l = local.read_text()
a = l.index("      // Randomly select the next point")
b = l.index("\n  void LocalCoveragePlanner::SelectViewPointFromFrontierQueue", a)
part = l[a:b].replace("queue.size()", "queue_copy.size()").replace("queue[i]", "queue_copy[i]")
part = part.replace("      std::uniform_int_distribution<int> gen_next_queue_idx(0, sample_range - 1);", "      if (sample_range <= 0) break;\n      std::uniform_int_distribution<int> gen_next_queue_idx(0, sample_range - 1);")
l = l[:a] + part + l[b:]
l = l.replace("    std::random_device rd;\n", "    if (sample_range <= 0) return;\n    std::random_device rd;\n", 1)
l = l.replace("      // Randomly select the next point", "      // Sample the current shrinking queue, not the original queue. Using its\n      // old length can select past end() after an erase and corrupt the heap.\n      // Randomly select the next point", 1)
local.write_text(l)
# Jazzy's .hpp cv_bridge include is .h in the Humble binary distribution.
for f in p.rglob("*"):
    if f.suffix in {".h", ".cpp"}:
        content = f.read_text()
        if "#include <cv_bridge/cv_bridge.hpp>" in content:
            f.write_text(content.replace("#include <cv_bridge/cv_bridge.hpp>",
                "#if __has_include(<cv_bridge/cv_bridge.hpp>)\n#include <cv_bridge/cv_bridge.hpp>\n#else\n#include <cv_bridge/cv_bridge.h>\n#endif"))
