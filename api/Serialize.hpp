#ifndef SERIALIZE_HPP
#define SERIALIZE_HPP

#include <string>

#include "Definitions.hpp"
#include "PrivateKey.hpp"
#include "PublicKey.hpp"

namespace fideslib {
enum SerType { BINARY = 0, JSON = 1 };
}

namespace fideslib::Serial {
bool SerializeToFile(const std::string& filename, const fideslib::CryptoContext<fideslib::DCRTPoly>& obj, const SerType& sertype);
bool SerializeToFile(const std::string& filename, const fideslib::PublicKey<fideslib::DCRTPoly>& obj, const SerType& sertype);
bool SerializeToFile(const std::string& filename, const fideslib::PrivateKey<fideslib::DCRTPoly>& obj, const SerType& sertype);

bool DeserializeFromFile(const std::string& filename, fideslib::CryptoContext<fideslib::DCRTPoly>& obj, const SerType& sertype);
bool DeserializeFromFile(const std::string& filename, fideslib::PublicKey<fideslib::DCRTPoly>& obj, const SerType& sertype);
bool DeserializeFromFile(const std::string& filename, fideslib::PrivateKey<fideslib::DCRTPoly>& obj, const SerType& sertype);
} // namespace fideslib::Serial

#endif // SERIALIZE_HPP