# ops-tensor Examples — mat_mul Operator (L2)
#
# Scenario aggregation: add_subdirectory each scenario.
# Add new scenarios here: add_subdirectory(mat_mul_swat)

add_subdirectory(mat_mul_streamk)
add_subdirectory(mat_mul_basic)
add_subdirectory(mat_mul_a_fullload)
add_subdirectory(mat_mul_b_fullload)
add_subdirectory(mat_mul_fixpipe_opti)
add_subdirectory(mat_mul_bmm_broadcast)
add_subdirectory(mat_mul_iterbatch_broadcast)
