
#include <cstdint>
#include <cstdio>
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
    auto train_x = load_csv_data("mnist_training_features.csv", 28, 28, 1);
    auto train_y = load_csv_data("mnist_training_labels.csv", 10, 1, 1);

    auto val_x = load_csv_data("mnist_validation_features.csv", 28, 28, 1);
    auto val_y = load_csv_data("mnist_validation_labels.csv", 10, 1, 1);

    vector<layer_t *> layers;

    conv_layer_t *layer1 = new conv_layer_t(1, 5, 8, train_x[0].size);        // 28 * 28 * 1 -> 24 * 24 * 8
    relu_layer_t *layer2 = new relu_layer_t(layer1->out.size);
    pool_layer_t *layer3 = new pool_layer_t(2, 2, layer2->out.size);                // 24 * 24 * 8 -> 12 * 12 * 8
    fc_layer_t *layer4 = new fc_layer_t(layer3->out.size, 10);                    // 4 * 4 * 16 -> 10

    layers.push_back((layer_t *) layer1);
    layers.push_back((layer_t *) layer2);
    layers.push_back((layer_t *) layer3);
    layers.push_back((layer_t *) layer4);

    float amse = 0;
    int ic = 0;

    for (long ep = 0; ep < 100000;) {

        for (int i = 0; i < train_x.size(); ++i) {

            auto xi = train_x[i];
            auto yi = train_y[i];

            float xerr = train(layers, xi, yi);

            amse += xerr;

            ep++;
            ic++;

            if (ep % 10000 == 0) {
                cout << "case " << ep << " err=" << amse / ic << endl;
                cout << "accuracy:" << compute_accuracy(layers, val_x, val_y) << endl;
                cout << "mae:" << compute_mae_loss(layers, val_x, val_y) << endl;
            }
        }
    }


#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-noreturn"
    while (true) {
        uint8_t *data = read_file("test.ppm");

        if (data) {
            uint8_t *usable = data;

            while (*(uint32_t *) usable != 0x0A353532)
                usable++;

#pragma pack(push, 1)
            struct RGB {
                uint8_t r, g, b;
            };
#pragma pack(pop)

            RGB *rgb = (RGB *) usable;

            tensor_t<float> image(28, 28, 1);
            for (int i = 0; i < 28; i++) {
                for (int j = 0; j < 28; j++) {
                    RGB rgb_ij = rgb[i * 28 + j];
                    image(j, i, 0) = (((float) rgb_ij.r
                                       + rgb_ij.g
                                       + rgb_ij.b)
                                      / (3.0f * 255.f));
                }
            }

            forward(layers, image);
            tensor_t<float> &out = layers.back()->out;
            for (int i = 0; i < 10; i++) {
                printf("[%i] %f\n", i, out(i, 0, 0) * 100.0f);
            }

            delete[] data;
        }

        struct timespec wait;
        wait.tv_sec = 1;
        wait.tv_nsec = 0;
        nanosleep(&wait, nullptr);
    }
#pragma clang diagnostic pop
    return 0;
}
