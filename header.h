#define N 10

typedef struct tDados {
    double data_x[N];
    double data_y[N];
    double data_z[N][N];
    double weight_x;
    double weight_y;
    double bias;
    double lr;
} Dados;

void carregar_dados(Dados& dados);
void testar_dados(Dados& dados);
void treinar_erro_quadratico_medio(Dados& dados);
void treinar_batch(Dados& dados);
float calcular_media(double* vet, int size);
float calcular_variancia(double* vet, int size);
float calcular_desvio_padrao(double* vet, int size);