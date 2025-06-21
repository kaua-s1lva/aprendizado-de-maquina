/**
 * Treinamento para uma função linear de apenas um parâmetro
 */

#include <stdio.h>
#include "header.h"
#include <math.h>

float calcular_funcao(double x, double y) {
    return (x * 1.23) + (y * 0) + 1.19;
}

void converter_valores_y(Dados& dados) {
    double media, desvio;
    media = calcular_media(dados.data_y, N);
    desvio = calcular_desvio_padrao(dados.data_y, N);

    for (int i=0; i<N; i++) {
        dados.data_y[i] = (dados.data_y[i] - media) / desvio;
    }
}
/*
void converter_valores_x(Dados& dados) {
    double media, desvio;
    media = calcular_media(dados.data_x, N);
    desvio = calcular_desvio_padrao(dados.data_x, N);

    for (int i=0; i<N; i++) {
        dados.data_x[i] = (dados.data_x[i] - media) / desvio;
        dados.data_y[i] = calcular_funcao(dados.data_x[i]);
    }
}
*/
void normalizar_dados(Dados& dados) {
    //normalizar x
    double media, desvio;
    media = calcular_media(dados.data_x, N);
    desvio = calcular_desvio_padrao(dados.data_x, N);

    for (int i=0; i<N; i++) {
        dados.data_x[i] = (dados.data_x[i] - media) / desvio;
    }

    //normalizar y
    media = calcular_media(dados.data_y, N);
    desvio = calcular_desvio_padrao(dados.data_y, N);

    for (int i=0; i<N; i++) {
        dados.data_y[i] = (dados.data_y[i] - media) / desvio;
    }

    //calcular z
    for (int i=0; i<N; i++) {
        for (int j=0; j<N; j++) {
            dados.data_z[i][j] = calcular_funcao(dados.data_x[i], dados.data_y[j]);
        }
    }
}

int main() {
    Dados dados;
    dados.weight_x = 0;
    dados.weight_y = 0;
    dados.bias = 0;
    dados.lr = 0.1;

    carregar_dados(dados);

    normalizar_dados(dados);
    //testar_dados(dados);

    treinar_erro_quadratico_medio(dados);

    return 0;
}

void carregar_dados(Dados& dados) {
    for (int i=0; i<N; i++) {
        dados.data_x[i] = i;
        dados.data_y[i] = i;
    }

    for (int i=0; i<N; i++) {
        for (int j=0; j<N; j++) {
            dados.data_z[i][j] = calcular_funcao(dados.data_x[i], dados.data_y[j]);
        }
    }
}

void testar_dados(Dados& dados) {
    printf("\nValores de x: \n");
    for (int i=0; i<N; i++) {
        printf(" %.2f ", dados.data_x[i]);
    }

    printf("\nValores de y: \n");
    for (int i=0; i<N; i++) {
        printf(" %.2f ", dados.data_y[i]);
    }

    printf("\nValores de z: \n");
    for (int i=0; i<N; i++) {
        for (int j=0; j<N; j++) {
            printf(" %6.2f ", dados.data_z[i][j]);
        }
        printf("\n");
    }
}

void treinar_erro_quadratico_medio(Dados& dados) {
    double x, y, error, z_pred, z_true;
    for (int epoca = 0; epoca < 10; epoca++) {
        for (int i=0; i < N; i++) {
            for (int j=0; j<N; j++) {
                x = dados.data_x[i];
                y = dados.data_y[j];
                z_true = dados.data_z[i][j];
    
                z_pred = (dados.weight_x * x) + (dados.weight_y * y) + dados.bias;
    
                error  = z_pred - z_true;          // ERROR = PREVISÃO – REAL
    
                double grad_wx = 2 * x * error;     // ∂/∂w MSE = (2)(x)(y_pred - y_true)
                double grad_wy = 2 * y * error;
                double grad_b = 2 * error;         // ∂/∂b MSE = (2)(y_pred - y_true)
    
                dados.weight_x -= dados.lr * grad_wx; // peso = peso - lr * grad_w
                dados.weight_y -= dados.lr * grad_wy;
                dados.bias   -= dados.lr * grad_b; // bias = bias - lr * grad_b
    
                printf("\nAprendido: z = %.4f * X + %.4f * Y + %.4f", dados.weight_x, dados.weight_y, dados.bias);
            }
        }
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