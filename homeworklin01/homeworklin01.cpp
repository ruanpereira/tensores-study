#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;

int main () {
    // definir um conjunto de vetores em um espaço vetorial tridimensional
    mat A = {{1,2,3},
             {4,5,6},
             {7,8,9}};

    cout << "hello" << A << endl;
    // verificar se os vetores sao linearmente independentes
    double det_A = det(A);
    // no caso, se essa operação der errado, os vetores não sao linearmente independentes
    cout << "det A = " << det_A << endl;
    // selecionar um subconjunto dos vetores e verificar se os vetores formam um subespaço vetorial
    // para isso, selecionamos um dos vetores. o conjunto são os vetores tridimensionais (x,y,z)
    // verificação se dá por: adicionando qualquer vetor no subspaço, x + y está no subspaço
    mat B = {{3,2,1}, {0,0,0}, {0,0,0}};
    cout << "Resultado de uma adição de vetor: " << A + B << endl;
    // o que nos mostra que ainda sim está no subspaço
    // verificação por multiplicação escalar
    mat C = A * 2;
    cout << "Resultado de uma multiplicação de escalar: " << C << endl; 
    uword r = arma::rank(A);
    cout << "Resultado das dimensões do subespaço: " << r << endl;

    //a base do subespaço é dado por um conjunto de vetores coluna que podem criar a matriz original a partir de 
    //uma operação matricial, e que não pode ser simplificado. 
    //ou seja, para a matriz A, podemos remontar tal matriz a partir de:
    mat base = {{1,0,0}, {0,1,0},{0,0,1}};
    vector<double> arb = {2,2,1};
    //podemos definir um vetor arbitrário como sendo por exemplo (2, 2, 1).
    //realizando a operação Ax = B usando a base encontrada anteriormente, podemos notar que o vetor resultante será sim do mesmo
    //as coordenadas seriam os vetores unitários dado pelo vetor arbitrário.
    // leitura do espaço linha, coluna, e nulo
    return 0;
}
