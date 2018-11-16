#include <stdio.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <thread>

#include "params/fc1_b.txt"
#include "params/fc1_w.txt"
#include "params/fc2_b.txt"
#include "params/fc2_w.txt"
#include "params/fc3_b.txt"
#include "params/fc3_w.txt"
#include "params/I_b.txt"
#include "params/I_w.txt"
#include "params/rnn1_bi.txt"
#include "params/rnn1_bh.txt"
#include "params/rnn1_wi.txt"
#include "params/rnn1_wh.txt"
#include "params/rnn2_bi.txt"
#include "params/rnn2_bh.txt"
#include "params/rnn2_wi.txt"
#include "params/rnn2_wh.txt"

#define SAMPLE_SIZE 83600
#define HIDDEN_SIZE 512
#define MELS_DIM 80
#define AUX_DIM 32

double mels[SAMPLE_SIZE][MELS_DIM];
double aux_0[SAMPLE_SIZE][AUX_DIM];
double aux_1[SAMPLE_SIZE][AUX_DIM];
double aux_2[SAMPLE_SIZE][AUX_DIM];
double aux_3[SAMPLE_SIZE][AUX_DIM];

using namespace std;

enum Layer {
    layer_I,
    layer_fc1,
    layer_fc2,
    layer_fc3,
    layer_rnn1,
    layer_rnn1_i,
    layer_rnn1_h,
    layer_rnn2,
    layer_rnn2_i,
    layer_rnn2_h
};

void loadfile() {
    // mels
    thread t1([]() {
        FILE *fp_mels;
        fp_mels = fopen("mels.txt", "r");
        for (int i = 0; i < SAMPLE_SIZE; i++) {
            for (int j = 0; j < MELS_DIM; j++) {
                fscanf(fp_mels, "%lf", &mels[i][j]);
            }
        }
        fclose(fp_mels);
    });

    // aux_0
    thread t2([]() {
        FILE *fp_aux_0;
        fp_aux_0 = fopen("aux_0.txt", "r");
        for (int i = 0; i < SAMPLE_SIZE; i++) {
            for (int j = 0; j < AUX_DIM; j++) {
                fscanf(fp_aux_0, "%lf", &aux_0[i][j]);
            }
        }
        fclose(fp_aux_0);
    });

    // aux_1
    thread t3([]() {
        FILE *fp_aux_1;
        fp_aux_1 = fopen("aux_1.txt", "r");
        for (int i = 0; i < SAMPLE_SIZE; i++) {
            for (int j = 0; j < AUX_DIM; j++) {
                fscanf(fp_aux_1, "%lf", &aux_1[i][j]);
            }
        }
        fclose(fp_aux_1);
    });

    // aux_2
    thread t4([]() {
        FILE *fp_aux_2;
        fp_aux_2 = fopen("aux_2.txt", "r");
        for (int i = 0; i < SAMPLE_SIZE; i++) {
            for (int j = 0; j < AUX_DIM; j++) {
                fscanf(fp_aux_2, "%lf", &aux_2[i][j]);
            }
        }
        fclose(fp_aux_2);
    });

    // aux_3
    thread t5([]() {
        FILE *fp_aux_3;
        fp_aux_3 = fopen("aux_3.txt", "r");
        for (int i = 0; i < SAMPLE_SIZE; i++) {
            for (int j = 0; j < AUX_DIM; j++) {
                fscanf(fp_aux_3, "%lf", &aux_3[i][j]);
            }
        }
        fclose(fp_aux_3);
    });

    t1.join();
    t2.join();
    t3.join();
    t4.join();
    t5.join();
}

template<class T> void savefile(T x[], int n) {
    FILE *fp;
    fp = fopen("output.txt", "w");
    for (int i = 0; i < n; i++) {
        fprintf(fp, "%.16f\n", x[i]);
    }
    fclose(fp);
}

template<class T> void debug(string tag, T x[], int n) {
    T sum = 0;
    T min = 0;
    T max = 0;
    for (int i = 0; i < n; i++) {
        sum += x[i];
        if (min > x[i]) {
            min = x[i];
        }
        if (max < x[i]) {
            max = x[i];
        }
    }

    T avg = sum / n;

    T var = 0;
    for (int i = 0; i < n; i++) {
        var += x[i] * x[i];
    }
    var = var / n - avg * avg;

    printf("[%s] min:%lf, max:%lf, avg:%lf, var:%lf\n", tag.c_str(), min, max, avg, var);
}

template<class T> void relu(T x[], int n) {
    T zero = 0;
    for (int i = 0; i < n; i++) {
        if (x[i] < 0) {
            x[i] = zero;
        }
    }
}

template<class T> void sigmoid(T x[], int n) {
    for (int i = 0; i < n; i++) {
        x[i] = 1.0 / (1.0 + exp(-x[i])); // TODO: exp tp pure c++
    }
}

template<class T> void softmax(T x[], int n) {
    T sum = 0;
    for (int i = 0; i < n; i++) {
        sum += exp(x[i]); // TODO: exp tp pure c++
    }

    for (int i = 0; i < n; i++) {
        x[i] = exp(x[i]) / sum;
    }
}

template<class T> void tanh(T x[], int n) {
    for (int i = 0; i < n; i++) {
        x[i] = tanh(x[i]); // TODO: tanh to pure C++
    }
}

template<class T> int choice(T x[], int n) {
    random_device rnd;
    mt19937 mt(rnd());
    uniform_real_distribution<double> dist(0.0, 1.0);
    double threshold = dist(mt);
    int id = 0;
    for (int i = 0; i < n; i++) {
        if (threshold < x[i]) {
            id = i;
            break;
        }
        threshold -= x[i];
    }
    return id;
}

template<class T> void concat(T out[], T x[], int start, int end) {
    for (int i = start; i < end; i++) {
        out[i] = x[i - start];
    }
}

template<class T> void add(T out[], T h[], int n) {
    for (int i = 0; i < n; i++) {
        out[i] += h[i];
    }
}

template<class T> void linear(T out[], T x[], Layer layer) {
    T sum;
    switch (layer) {
    case layer_I:
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            sum = 0;
            for (int j = 0; j < 1 + MELS_DIM + AUX_DIM; j++) {
                sum += x[j] * I_w[i][j];
            }
            out[i] = I_b[i] + sum;
        }
        break;
    case layer_fc1:
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            sum = 0;
            for (int j = 0; j < HIDDEN_SIZE + AUX_DIM; j++) {
                sum += x[j] * fc1_w[i][j];
            }
            out[i] = fc1_b[i] + sum;
        }
        break;
    case layer_fc2:
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            sum = 0;
            for (int j = 0; j < HIDDEN_SIZE + AUX_DIM; j++) {
                sum += x[j] * fc2_w[i][j];
            }
            out[i] = fc2_b[i] + sum;
        }
        break;
    case layer_fc3:
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            sum = 0;
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                sum += x[j] * fc3_w[i][j];
            }
            out[i] = fc3_b[i] + sum;
        }
        break;
    case layer_rnn1_i:
        for (int i = 0; i < HIDDEN_SIZE * 3; i++) {
            sum = 0;
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                sum += x[j] * rnn1_wi[i][j];
            }
            out[i] = rnn1_bi[i] + sum;
        }
        break;
    case layer_rnn1_h:
        for (int i = 0; i < HIDDEN_SIZE * 3; i++) {
            sum = 0;
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                sum += x[j] * rnn1_wh[i][j];
            }
            out[i] = rnn1_bh[i] + sum;
        }
        break;
    case layer_rnn2_i:
        for (int i = 0; i < HIDDEN_SIZE * 3; i++) {
            sum = 0;
            for (int j = 0; j < HIDDEN_SIZE + AUX_DIM; j++) {
                sum += x[j] * rnn2_wi[i][j];
            }
            out[i] = rnn2_bi[i] + sum;
        }
        break;
    case layer_rnn2_h:
        for (int i = 0; i < HIDDEN_SIZE * 3; i++) {
            sum = 0;
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                sum += x[j] * rnn2_wh[i][j];
            }
            out[i] = rnn2_bh[i] + sum;
        }
        break;
    }
}

template<class T> void gru(T x[], T h[], Layer layer) {
    T igates[HIDDEN_SIZE * 3];
    T hgates[HIDDEN_SIZE * 3];
    T reset_gate[HIDDEN_SIZE];
    T input_gate[HIDDEN_SIZE];
    T new_gate[HIDDEN_SIZE];

    // igates, hgates
    switch (layer) {
    case layer_rnn1:
        linear(igates, x, layer_rnn1_i);
        linear(hgates, h, layer_rnn1_h);
        break;
    case layer_rnn2:
        linear(igates, x, layer_rnn2_i);
        linear(hgates, h, layer_rnn2_h);
        break;
    }

    // reset_gate, input_gate
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        reset_gate[i] = igates[i] + hgates[i];
        input_gate[i] = igates[HIDDEN_SIZE + i] + hgates[HIDDEN_SIZE + i];
    }
    sigmoid(reset_gate, HIDDEN_SIZE);
    sigmoid(input_gate, HIDDEN_SIZE);

    // new_gate
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        new_gate[i] = igates[HIDDEN_SIZE * 2 + i] + reset_gate[i] * hgates[HIDDEN_SIZE * 2 + i];
    }
    tanh(new_gate, HIDDEN_SIZE);

    // h_next
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        h[i] = new_gate[i] + input_gate[i] * (h[i] - new_gate[i]);
    }
}

int main() {
    double out[SAMPLE_SIZE];
    double I[1 + MELS_DIM + AUX_DIM];
    double inp[HIDDEN_SIZE + AUX_DIM];
    double x[HIDDEN_SIZE];
    double p[HIDDEN_SIZE];
    double h1[HIDDEN_SIZE] = {};
    double h2[HIDDEN_SIZE] = {};
    double sample[1] = {};

    printf("***** Start WaveRNN inference *****\n");

    // load inputs
    printf("Loading inputs from file...\n");
    loadfile();

    // inference loop
    printf("Enter inference loop!!!\n");
    auto start = chrono::system_clock::now();
    for (int i = 0; i < SAMPLE_SIZE; i++) {
        // I
        concat(I, sample, 0, 1);
        concat(I, mels[i], 1, 1 + MELS_DIM);
        concat(I, aux_0[i], 1 + MELS_DIM, 1 + MELS_DIM + AUX_DIM);
        linear(x, I, layer_I);

        // rnn1
        gru(x, h1, layer_rnn1);

        // rnn2
        add(x, h1, HIDDEN_SIZE);
        concat(inp, x, 0, HIDDEN_SIZE);
        concat(inp, aux_1[i], HIDDEN_SIZE, HIDDEN_SIZE + AUX_DIM);
        gru(inp, h2, layer_rnn2);

        // fc1
        add(x, h2, HIDDEN_SIZE);
        concat(inp, x, 0, HIDDEN_SIZE);
        concat(inp, aux_2[i], HIDDEN_SIZE, HIDDEN_SIZE + AUX_DIM);
        linear(x, inp, layer_fc1);
        relu(x, HIDDEN_SIZE);

        // fc2
        concat(inp, x, 0, HIDDEN_SIZE);
        concat(inp, aux_3[i], HIDDEN_SIZE, HIDDEN_SIZE + AUX_DIM);
        linear(x, inp, layer_fc2);
        relu(x, HIDDEN_SIZE);

        // fc3
        linear(p, x, layer_fc3);

        // categorize
        softmax(p, HIDDEN_SIZE);
        sample[0] = 2 * choice(p, HIDDEN_SIZE) / (HIDDEN_SIZE - 1.) - 1.;
        out[i] = sample[0];

        // show progress
        if (((i + 1) % (SAMPLE_SIZE / 10)) == 0) {
            auto end = chrono::system_clock::now();
            auto sec = chrono::duration<double>(end - start).count();
            printf("|%7.2lf s||%3d %%|", sec, ((i + 1) / (SAMPLE_SIZE / 100)));
            for (int j = 0; j < ((i + 1) / (SAMPLE_SIZE / 10)); j++) {
                printf("##");
            }
            printf("\n");
        }
    }

    // save outputs
    printf("Saving outputs to file...\n");
    savefile(out, SAMPLE_SIZE);

    printf("***** Finish WaveRNN inference *****\n");

    return 0;
}
