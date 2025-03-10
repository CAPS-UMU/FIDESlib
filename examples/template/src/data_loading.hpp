#ifndef DATA_LOADING_HPP
#define DATA_LOADING_HPP

#include <string>
#include <vector>

/**
 * Carga fichero CSV en memoria.
 * @param filename Ruta al fichero CSV.
 * @param data Matriz de reales de doble precisión donde se almacena el resultado.
 * @return Verdadero si se tubo éxito.
 * @note Cada vector de data se corresponde con una línea del fichero CSV.
 */
bool load_csv(const std::string &filename, std::vector<std::vector<double> > &data);


#endif //DATA_LOADING_HPP
