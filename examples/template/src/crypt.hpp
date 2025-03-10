#ifndef CRYPT_HPP
#define CRYPT_HPP

#include <vector>

#include <openfhe.h>

/**
 * Encripta una matriz de datos.
 * @param data Datos que se desean encriptar.
 * @param cc Contexto de OpenFHE utilizado para encriptar los datos.
 * @param pk Clave pública utilizada para encriptar los datos.
 * @return Vector que contiene un texto cifrado para cada fila de la matriz.
 */
std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly> > encrypt_data(const std::vector<std::vector<double> > &data,
                                                                    const lbcrypto::CryptoContext<lbcrypto::DCRTPoly> &
                                                                    cc,
                                                                    const lbcrypto::PublicKey<lbcrypto::DCRTPoly> &pk);

/**
 * Encripta un vector de datos.
 * @param data Datos que se desean encriptar.
 * @param cc Contexto de OpenFHE utilizado para encriptar los datos.
 * @param pk Clave pública utilizada pra encriptar los datos.
 * @return Texto cifrado con el vector de datos encriptado.
 */
lbcrypto::Ciphertext<lbcrypto::DCRTPoly> encrypt_data(std::vector<double> &data,
													  const lbcrypto::CryptoContext<lbcrypto::DCRTPoly> &cc,
													  const lbcrypto::PublicKey<lbcrypto::DCRTPoly> &pk);

#endif //CRYPT_H
