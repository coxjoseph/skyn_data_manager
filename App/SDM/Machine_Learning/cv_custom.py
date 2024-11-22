def group_k_fold_cross_validation(estimator, features, X, y):
    cv_results = {
        'subid': [],
        'y_truth': [],
        'predictions': [],
        'predicted_prob': [],
        'prediction_correct': [],
        'prediction_result': []
    }
    cv_stats = {
        'pred_count': 0,
        'folds': 0,
        'correct': 0,
        'accuracy': 0.0,
        'auc_roc': 'NA',
        'TP': 0,
        'TN': 0,
        'FP': 0,
        'FN': 0
    }
    # occasions_to_assess = [occasion for occasion in cohort_processor.occasions if occasion.subid in ]
    for subid in features['subid'].unique().tolist():
        # the split
        x_train = X[features['subid'] != subid]
        y_train = y[features['subid'] != subid]
        x_test = X[features['subid'] == subid]
        y_test = y[features['subid'] == subid].tolist()

        # train
        fitted = estimator.fit(x_train, y_train)

        # results updates
        cv_results["subid"].append(subid)
        cv_results['y_truth'].append(y_test)
        predictions = fitted.predict(x_test)
        cv_results['predictions'].append(predictions)
        probabilities = fitted.predict_proba(x_test)
        cv_results['predicted_prob'].append(probabilities)
        correct_predictions = [y_test[i] == predictions[i] for i in range(0, len(predictions))]
        cv_results['prediction_correct'].append(correct_predictions)

        tps = [((y_test[i] == predictions[i]) and (predictions[i] == 1)) for i in range(0, len(predictions))]
        tns = [((y_test[i] == predictions[i]) and (predictions[i] == 0)) for i in range(0, len(predictions))]
        fps = [((y_test[i] != predictions[i]) and (predictions[i] == 1)) for i in range(0, len(predictions))]
        fns = [((y_test[i] != predictions[i]) and (predictions[i] == 0)) for i in range(0, len(predictions))]

        cv_results['prediction_result'].append(
            [['TP', 'TN', 'FP', 'FN'][i] for i, result in enumerate([tps, tns, fps, fns]) if any(result)])
        # stat updates
        cv_stats['folds'] += 1
        cv_stats['pred_count'] += len(y_test)
        cv_stats['correct'] += sum(correct_predictions)
        cv_stats['accuracy'] = cv_stats['correct'] / cv_stats['pred_count']
        cv_stats['TP'] += sum(tps)
        cv_stats['TN'] += sum(tns)
        cv_stats['FP'] += sum(fps)
        cv_stats['FN'] += sum(fns)

    return cv_results, cv_stats
