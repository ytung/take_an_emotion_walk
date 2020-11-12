import numpy as np
import pickle
from sklearn.mixture import GaussianMixture
from utils import loader


def main():
    num_joints = 21
    num_labels = [4, 3, 3]
    num_classes = 4

    data, labels, [data_train, data_test, labels_train, labels_test] = \
        loader.load_edin_data(
            'datasets/data_edin_locomotion_pose_diff_aff_drop_5.npz',
            'datasets/labels_edin_locomotion', num_labels)

    model1 = GaussianMixture(n_components=num_joints, covariance_type='full', random_state=1, max_iter=200)
    model2 = GaussianMixture(n_components=num_joints, covariance_type='full', random_state=1, max_iter=200)
    model3 = GaussianMixture(n_components=num_joints, covariance_type='full', random_state=1, max_iter=200)
    model4 = GaussianMixture(n_components=num_joints, covariance_type='full', random_state=1, max_iter=200)
    models = [model1, model2, model3, model4]

    train_samples, frames, joints, dim = data_train.shape
    data_train = data_train.reshape(train_samples * frames, joints * dim)
    data_test = data_test.reshape(data_test.shape[0] * frames, joints * dim)
    labels_train = np.repeat(labels_train, frames, axis=0)
    labels_test = np.repeat(labels_test, frames, axis=0)

    # for i in range(len(models)):
    #     model = models[i]
    #     indices = np.argmax(labels_train, axis=1) == i
    #     print(len(indices), data_train[indices].shape)
    #     model.fit(data_train[indices], )

    models = []
    for i in range(num_classes):
        model = pickle.load(open("model%d.pkl" % i, "rb"))
        models.append(model)

    # for i in range(len(models)):
    #     pickle.dump(models[i], open("model%d.pkl" % i, "wb"))

    predictions = []
    for model in models:
        pred = model.score_samples(data_test)
        predictions.append(pred.reshape(pred.shape[0], -1))

    preds = np.concatenate(predictions, axis=1)
    pred_classes = np.argmax(preds, axis=1)
    actual_classes = np.argmax(labels_test, axis=1)
    compare = pred_classes == actual_classes

    print("Accuracy: %f" % (np.count_nonzero(compare)/float(len(compare))))

    for i in range(num_classes):
        indices = pred_classes == i
        actual = actual_classes[indices]
        precision = np.count_nonzero(actual == i) / float(len(actual))
        print("Class %d precision %f" % (i, precision))


if __name__ == "__main__":
    main()
