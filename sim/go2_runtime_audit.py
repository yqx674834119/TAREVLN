"""Read actual PhysX parameters, including properties of instanceable shapes."""
def physics_snapshot(view):
    result={}
    for key in ('dof_stiffnesses','dof_dampings','dof_max_forces','dof_friction_coefficients',
                'dof_max_velocities','dof_armatures','masses','coms','inertias','material_properties'):
        result[key]=getattr(view,'get_'+key)().tolist()
    return result
