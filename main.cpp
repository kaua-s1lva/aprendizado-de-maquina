/**
 * treinado por gradiente descendente para aproximar a função y = 2x.
 */

#include <stdio.h>
#include "header.h"
#include <math.h>

void converter_valores_y(Dados& dados) {
    double media, desvio;
    media = calcular_media(dados.data_y, N);
    desvio = calcular_desvio_padrao(dados.data_y, N);

    for (int i=0; i<N; i++) {
        dados.data_y[i] = (dados.data_y[i] - media) / desvio;
    }
}

void converter_valores_x(Dados& dados) {
    double media, desvio;
    media = calcular_media(dados.data_x, N);
    desvio = calcular_desvio_padrao(dados.data_x, N);

    for (int i=0; i<N; i++) {
        dados.data_x[i] = (dados.data_x[i] - media) / desvio;
        dados.data_y[i] = dados.data_x[i] * 2;
    }
}

int main() {
    Dados dados;
    dados.weight = 0;
    dados.bias = 0;
    dados.lr = 0.1;

    carregar_dados(dados);

    //testar_dados(dados);

    converter_valores_x(dados);

    treinar_erro_quadratico_medio(dados);

    return 0;
}

void carregar_dados(Dados& dados) {
    for (int i=0; i<N; i++) {
        dados.data_x[i] = i;
        dados.data_y[i] = 2 * i;
    }
}

void testar_dados(Dados& dados) {
    printf("\nValores de x: \n");
    for (int i=0; i<N; i++) {
        printf(" %.0f ", dados.data_x[i]);
    }

    printf("\nValores de y: \n");
    for (int i=0; i<N; i++) {
        printf(" %.0f ", dados.data_y[i]);
    }
}

void treinar_erro_quadratico_medio(Dados& dados) {
    double x, y_true, y_pred, error, gradiente;
    for (int epoca = 0; epoca < 1; epoca++) {
        for (int i=0; i < N; i++) {
            x = dados.data_x[i];
            y_true = dados.data_y[i];

            y_pred = dados.weight * x + dados.bias;

            error = y_pred - y_true;

            //if (i == 19) printf("\n[%d]{peso: %.4f, x: %.4f, y: %.4f}", i, dados.weight, x, y_true);
            gradiente = (-2 * x) * (y_true - dados.weight * x - dados.bias);
            //gradiente = -2 * x * error;
            dados.weight = dados.weight - (dados.lr * gradiente);
            dados.bias = dados.lr * error;
            printf("\nAprendido: y = %.4f * X + %.4f", dados.weight, dados.bias);
            //printf("\n[%d]: %.4f", i, gradiente);
        }
    }
}

void treinar_batch(Dados& dados) {
    double x, y_true, y_pred, error, gradiente=0;
    for (int epoca = 0; epoca < 1; epoca++) {
        for (int i=0; i < N; i++) {
            x = dados.data_x[i];

            y_true = dados.data_y[i];

            y_pred = dados.weight * x + dados.bias;

            error += y_pred - y_true;

            gradiente += (-2 * x) * (y_true - dados.weight * x);

        }
        dados.weight = dados.weight - (dados.lr * (gradiente / N));

        dados.bias = dados.lr * error / N;

        printf("\nAprendido: y = %.4f * X + %.4f", dados.weight, dados.bias);
    }
}

float calcular_media(double* vet, int size) {
    double media=0;
    for (int i=0; i<size; i++) {
        media += vet[i];
    }
    return media / size;
}

float calcular_variancia(double* vet, int size) {
    double var=0, media=0;

    media = calcular_media(vet, size);

    for (int i=0; i<size; i++) {
        var += pow(vet[i] - media, 2);
    }
    var = var / (size - 1);
    return var;
}

float calcular_desvio_padrao(double* vet, int size) {
    return sqrt(calcular_variancia(vet, size));
}