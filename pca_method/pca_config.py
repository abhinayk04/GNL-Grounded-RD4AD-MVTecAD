
PCA_CONFIG = {
    "target_category": "bottle",

    "num_components": 16,

    "candidate_components": [8, 16, 32],

    "basis_type": "category_specific",

    "feature_difference": "teacher_original_minus_teacher_transformed",

    "inference_residual": "teacher_minus_student",

    "feature_layout": "BCHW",

    "projection": "orthogonal",

    "save_directory": "pca_method/saved_bases",

    "random_seed": 42,
}
