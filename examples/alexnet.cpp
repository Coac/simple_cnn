
#include <cstdint>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <time.h>
#include <sstream>
#include "CNN/cnn.h"

using namespace std;


void forward(vector<layer_t *> &layers, tensor_t<float> &data) {
    activate(layers[0], data);
    for (int i = 1; i < layers.size(); i++) {
        activate(layers[i], layers[i - 1]->out);
    }
}


float train(vector<layer_t *> &layers, tensor_t<float> &data, tensor_t<float> &expected) {
    forward(layers, data);

    tensor_t<float> grads = layers.back()->out - expected;

    for (int i = layers.size() - 1; i >= 0; i--) {
        if (i == layers.size() - 1)
            calc_grads(layers[i], grads);
        else
            calc_grads(layers[i], layers[i + 1]->grads_in);
    }

    for (int i = 0; i < layers.size(); i++) {
        fix_weights(layers[i]);
    }

    float err = 0;
    for (int i = 0; i < grads.size.x * grads.size.y * grads.size.z; i++) {
        float f = expected.data[i];
        if (f > 0.5)
            err += abs(grads.data[i]);
    }
    return err * 100;
}


uint8_t *read_file(const char *szFile) {
    ifstream file(szFile, ios::binary | ios::ate);
    streamsize size = file.tellg();
    file.seekg(0, ios::beg);

    if (size == -1)
        return nullptr;

    uint8_t *buffer = new uint8_t[size];
    file.read((char *) buffer, size);
    return buffer;
}

vector<vector<float>> load_csv(const char *csv_path) {
    vector<vector<float>> values;
    vector<float> valueline;
    ifstream fin(csv_path);
    if (fin.fail()) {
        cout << csv_path << " not found !" << endl;
        exit(1);
    }
    string item;
    for (string line; getline(fin, line);) {
        istringstream in(line);

        while (getline(in, item, ',')) {
            valueline.push_back(atof(item.c_str()));
        }

        values.push_back(valueline);
        valueline.clear();
    }

    cout << "Shape: (" << values.size() << "," << values[0].size() << ")" << endl;

    return values;
}


vector<tensor_t<float>>
csv_to_tensor(vector<vector<float>> &csv_y, const int size_x, const int size_y, const int size_z) {
    vector<tensor_t<float>> tensors_y;

    for (auto &yi : csv_y) {
        tensor_t<float> tensor_y(size_x, size_y, size_z);

        int i = 0;
        int j = 0;
        int k = 0;
        for (auto &yi_col : yi) {
            tensor_y(i, j, k) = yi_col;
            i++;
            if (i % size_x == 0) {
                i = 0;
                j++;

                if (j % size_y == 0) {
                    j = 0;
                    k++;
                }
            }
        }

        tensors_y.push_back(tensor_y);
    }

    return tensors_y;
}

vector<tensor_t<float>> load_csv_data(const char *csv_path, const int size_x, const int size_y, const int size_z) {
    auto csv_x = load_csv(csv_path);
    return csv_to_tensor(csv_x, size_x, size_y, size_z);
}

float compute_accuracy(vector<layer_t *> &layers, vector<tensor_t<float>> &x, vector<tensor_t<float>> &y) {

    float correct_count = 0;

    for (int i = 0; i < x.size(); ++i) {

        auto xi = x[i];
        auto yi = y[i];

        float expected = 0;
        for (int i = 0; i < yi.size.x; i++) {
            if (yi(i, 0, 0) == 1) {
                expected = i;
            }
        }


        forward(layers, xi);
        auto probs = layers.back()->out;
        float predicted = -1;
        float max_prob = -1;
        for (int i = 0; i < probs.size.x; i++) {
            if (probs(i, 0, 0) > max_prob) {
                max_prob = probs(i, 0, 0);
                predicted = i;
            }
        }

        if (predicted == expected) {
            correct_count++;
        }
    }

    return correct_count / x.size();
}

float compute_mae_loss(vector<layer_t *> &layers, vector<tensor_t<float>> &x, vector<tensor_t<float>> &y) {
    float sum = 0;

    for (int i = 0; i < x.size(); ++i) {

        auto xi = x[i];
        auto yi = y[i];

        forward(layers, xi);
        auto predicted_yi = layers.back()->out;

        auto diff = yi - predicted_yi;

        for (int i = 0; i < diff.size.x; i++) {
            sum += abs(diff(i, 0, 0));
        }
    }

    return sum / x.size();
}

int main() {
    cout << "Loading training set" << endl;
    auto train_x = load_csv_data("mnist_training_features.csv", 28, 28, 1);
    auto train_y = load_csv_data("mnist_training_labels.csv", 10, 1, 1);

    cout << "Loading validation set" << endl;
    auto val_x = load_csv_data("mnist_validation_features.csv", 28, 28, 1);
    auto val_y = load_csv_data("mnist_validation_labels.csv", 10, 1, 1);

    vector<layer_t *> layers;

    // Stage 1
    conv_layer_t *conv1 = new conv_layer_t(4, 11, 96, train_x[0].size);
    relu_layer_t *relu1 = new relu_layer_t(conv1->out.size);
    pool_layer_t *pool1 = new pool_layer_t(2, 3, relu1->out.size);

    // Stage 2
    conv_layer_t *conv2 = new conv_layer_t(1, 5, 256, pool1->out.size);
    relu_layer_t *relu2 = new relu_layer_t(conv2->out.size);
    pool_layer_t *pool2 = new pool_layer_t(2, 3, relu2->out.size);


    // Stage 3
    conv_layer_t *conv3 = new conv_layer_t(1, 3, 384, pool2->out.size);
    relu_layer_t *relu3 = new relu_layer_t(conv3->out.size);
    conv_layer_t *conv4 = new conv_layer_t(1, 3, 384, relu3->out.size);
    relu_layer_t *relu4 = new relu_layer_t(conv4->out.size);
    conv_layer_t *conv5 = new conv_layer_t(1, 3, 256, relu4->out.size);
    relu_layer_t *relu5 = new relu_layer_t(conv5->out.size);
    pool_layer_t *pool3 = new pool_layer_t(2, 3, relu5->out.size);

    // Stage 4
    fc_layer_t *fc1 = new fc_layer_t(pool3->out.size, 4096);
    relu_layer_t *relu6 = new relu_layer_t(fc1->out.size);
    dropout_layer_t *dropout1 = new dropout_layer_t(relu6->out.size, 0.5);

    // Stage 5
    fc_layer_t *fc2 = new fc_layer_t(dropout1->out.size, 4096);
    relu_layer_t *relu7 = new relu_layer_t(fc2->out.size);
    dropout_layer_t *dropout2 = new dropout_layer_t(relu7->out.size, 0.5);

    // Stage 6
    fc_layer_t *fc3 = new fc_layer_t(dropout2->out.size, val_y[0].size.x);
    // Softmax


    layers.push_back((layer_t *) conv1);
    layers.push_back((layer_t *) relu1);
    layers.push_back((layer_t *) pool1);

    layers.push_back((layer_t *) conv2);
    layers.push_back((layer_t *) relu2);
    layers.push_back((layer_t *) pool2);

    layers.push_back((layer_t *) conv3);
    layers.push_back((layer_t *) relu3);
    layers.push_back((layer_t *) conv4);
    layers.push_back((layer_t *) relu4);
    layers.push_back((layer_t *) conv5);
    layers.push_back((layer_t *) relu5);
    layers.push_back((layer_t *) pool3);

    layers.push_back((layer_t *) fc1);
    layers.push_back((layer_t *) relu6);
    layers.push_back((layer_t *) dropout1);

    layers.push_back((layer_t *) fc2);
    layers.push_back((layer_t *) relu7);
    layers.push_back((layer_t *) dropout2);

    layers.push_back((layer_t *) fc3);


    float amse = 0;
    int ic = 0;

    clock_t begin = clock();

    for (long ep = 0; ep < 100000;) {

        for (int i = 0; i < train_x.size(); ++i) {

            auto xi = train_x[i];
            auto yi = train_y[i];

            float xerr = train(layers, xi, yi);

            amse += xerr;

            ep++;
            ic++;

            if (ep % 1000 == 0) {
                auto val_acc = compute_accuracy(layers, val_x, val_y);
                auto val_loss = compute_mae_loss(layers, val_x, val_y);
                clock_t end = clock();
                double elapsed_time = double(end - begin) / CLOCKS_PER_SEC * 1000;
                cout << "eval:" << ep << "," << ep << ",0,0," << val_loss << "," << val_acc << "," << elapsed_time
                     << endl;
            }
        }
    }

    return 0;
}
