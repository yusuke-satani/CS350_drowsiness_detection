import multiprocessing
import argparse
import dlib
from sklearn.model_selection import ParameterGrid, KFold
import numpy as np

def train_and_evaluate(train_xml, test_xml, options):
    dlib.train_shape_predictor(train_xml, "temp_model.dat", options)
    error = dlib.test_shape_predictor(test_xml, "temp_model.dat")
    return error

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--training", required=True, help="path to input training XML file")
    ap.add_argument("-m", "--model", required=True, help="path serialized dlib shape predictor model")
    args = vars(ap.parse_args())

    # パラメータグリッドの定義
    param_grid = {
        'tree_depth': [3, 4, 5],
        'nu': [0.05, 0.1, 0.2],
        'cascade_depth': [10, 15, 20],
        'feature_pool_size': [300, 400, 500],
        'num_test_splits': [40, 50, 60],
        'oversampling_amount': [3, 5, 7],
        'oversampling_translation_jitter': [0.05, 0.1, 0.15]
    }

    # K-fold クロスバリデーションの設定
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    best_error = float('inf')
    best_params = None

    # パラメータグリッドの探索
    for params in ParameterGrid(param_grid):
        options = dlib.shape_predictor_training_options()
        options.tree_depth = params['tree_depth']
        options.nu = params['nu']
        options.cascade_depth = params['cascade_depth']
        options.feature_pool_size = params['feature_pool_size']
        options.num_test_splits = params['num_test_splits']
        options.oversampling_amount = params['oversampling_amount']
        options.oversampling_translation_jitter = params['oversampling_translation_jitter']
        options.be_verbose = False
        options.num_threads = multiprocessing.cpu_count()

        errors = []

        # クロスバリデーション
        for train_index, test_index in kf.split(open(args["training"]).readlines()):
            train_xml = f"train_fold.xml"
            test_xml = f"test_fold.xml"
            
            with open(train_xml, "w") as f:
                f.writelines([open(args["training"]).readlines()[i] for i in train_index])
            with open(test_xml, "w") as f:
                f.writelines([open(args["training"]).readlines()[i] for i in test_index])

            error = train_and_evaluate(train_xml, test_xml, options)
            errors.append(error)

        mean_error = np.mean(errors)
        print(f"Parameters: {params}, Mean Error: {mean_error}")

        if mean_error < best_error:
            best_error = mean_error
            best_params = params

    print("\nBest Parameters:")
    print(best_params)
    print(f"Best Mean Error: {best_error}")

    # 最適なパラメータでモデルを再トレーニング
    final_options = dlib.shape_predictor_training_options()
    for key, value in best_params.items():
        setattr(final_options, key, value)
    final_options.be_verbose = True
    final_options.num_threads = multiprocessing.cpu_count()

    print("\nTraining final model with best parameters...")
    dlib.train_shape_predictor(args["training"], args["model"], final_options)

if __name__ == "__main__":
    main()