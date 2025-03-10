#include "data_loading.hpp"
#include "context.hpp"

#include <iostream>
#include <vector>

#include <openfhe.h>

#include <CKKS/Context.cuh>
#include <LimbUtils.cuh>
#include <CKKS/openfhe-interface/RawCiphertext.cuh>
#include <CKKS/Ciphertext.cuh>
#include <CKKS/KeySwitchingKey.cuh>
#include <CKKS/Bootstrap.cuh>

std::vector<FIDESlib::PrimeRecord> p64{
    {.p = 2305843009218281473}, {.p = 2251799661248513}, {.p = 2251799661641729}, {.p = 2251799665180673},
    {.p = 2251799682088961},    {.p = 2251799678943233}, {.p = 2251799717609473}, {.p = 2251799710138369},
    {.p = 2251799708827649},    {.p = 2251799707385857}, {.p = 2251799713677313}, {.p = 2251799712366593},
    {.p = 2251799716691969},    {.p = 2251799714856961}, {.p = 2251799726522369}, {.p = 2251799726129153},
    {.p = 2251799747493889},    {.p = 2251799741857793}, {.p = 2251799740416001}, {.p = 2251799746707457},
    {.p = 2251799756013569},    {.p = 2251799775805441}, {.p = 2251799763091457}, {.p = 2251799767154689},
    {.p = 2251799765975041},    {.p = 2251799770562561}, {.p = 2251799769776129}, {.p = 2251799772266497},
    {.p = 2251799775281153},    {.p = 2251799774887937}, {.p = 2251799797432321}, {.p = 2251799787995137},
    {.p = 2251799787601921},    {.p = 2251799791403009}, {.p = 2251799789568001}, {.p = 2251799795466241},
    {.p = 2251799807131649},    {.p = 2251799806345217}, {.p = 2251799805165569}, {.p = 2251799813554177},
    {.p = 2251799809884161},    {.p = 2251799810670593}, {.p = 2251799818928129}, {.p = 2251799816568833},
    {.p = 2251799815520257}};

std::vector<FIDESlib::PrimeRecord> sp64{
    {.p = 2305843009218936833}, {.p = 2305843009220116481}, {.p = 2305843009221820417}, {.p = 2305843009224179713},
    {.p = 2305843009225228289}, {.p = 2305843009227980801}, {.p = 2305843009229160449}, {.p = 2305843009229946881},
    {.p = 2305843009231650817}, {.p = 2305843009235189761}, {.p = 2305843009240301569}, {.p = 2305843009242923009},
    {.p = 2305843009244889089}, {.p = 2305843009245413377}, {.p = 2305843009247641601}};

FIDESlib::CKKS::Parameters params{.logN = 13, .L = 5, .dnum = 2, .primes = p64, .Sprimes = sp64};

/// Operación de vectores normal y corriente.
void operaciones_normal(std::vector<double> &a, const std::vector<double> &b) {

	/// 1. Suma	
	for (int i = 0; i < a.size(); ++i) {
		a[i] += b[i];
	}

	/// 2. Multiplicación.
	for (int i = 0; i < a.size(); ++i) {
		a[i] *= b[i];
	}

	/// 3 Rotación.
	int indice_rotacion = 2;
	std::vector<double> tmp (a.size(), 0);
	for (unsigned int i = 0; i < a.size(); ++i) {
		tmp[i] = a[((i+indice_rotacion)%a.size())];
	}
	a = tmp;
}

/// Operaciones de vectores utilizando OpenFHE.
void operaciones_openfhe(std::vector<double> &a, const std::vector<double> &b) {

	/// 1. Generar contexto.
	lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc = generate_context();

	/// 2. Crear claves.
	auto keys = cc->KeyGen();
	/// 2.1 Crear clave de multiplicación en OpenFHE.
	cc->EvalMultKeyGen(keys.secretKey);
	/// 2.2 Crear claves de rotación en OpenFHE.
	std::vector<int> indices_rotacion {2, -1};
	cc->EvalRotateKeyGen(keys.secretKey, indices_rotacion);

	/// 3. Encriptar datos de entrada.
	lbcrypto::Plaintext a_texto_plano = cc->MakeCKKSPackedPlaintext(a);
	lbcrypto::Plaintext b_texto_plano = cc->MakeCKKSPackedPlaintext(b);
	lbcrypto::Ciphertext<lbcrypto::DCRTPoly> a_texto_cifrado = cc->Encrypt(keys.publicKey, a_texto_plano);
	lbcrypto::Ciphertext<lbcrypto::DCRTPoly> b_texto_cifrado = cc->Encrypt(keys.publicKey, b_texto_plano);

	/// 4. Operar con textos cifrados.
	/// 4.1 Suma.
	auto res_texto_cifrado = cc->EvalAdd(a_texto_cifrado, b_texto_cifrado);
	/// 4.2 Multiplicación.
	res_texto_cifrado = cc->EvalMult(res_texto_cifrado, b_texto_cifrado);
    /// 4.3 Rotación.
	int indice_rotacion = 2;
	res_texto_cifrado = cc->EvalRotate(res_texto_cifrado, indice_rotacion);

	/// 5. Desencriptar texto cifrado.
	lbcrypto::Plaintext res_texto_plano;
	cc->Decrypt(keys.secretKey, res_texto_cifrado, &res_texto_plano);
	res_texto_plano->SetLength(a.size());
	auto res = res_texto_plano->GetCKKSPackedValue();

	for (int i = 0; i < a.size(); ++i) {
		a[i] = res[i].real();
	}

	/// ATENCIÓN. Cuando no quieras usar más un contexto, usa estos métodos para ahorrar memoria.
	cc->ClearEvalMultKeys();
	cc->ClearEvalAutomorphismKeys();
}

/// Operaciones de vectores utilizando FIDESlib.
void operaciones_fideslib(std::vector<double> &a, const std::vector<double> &b) {
	
	/// 1. Generar contexto.
	lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc = generate_context();

	/// 2. Crear claves.
	auto keys = cc->KeyGen();
	/// 2.1 Crear clave de multiplicación en OpenFHE.
	cc->EvalMultKeyGen(keys.secretKey);
	/// 2.2 Crear claves de rotación en OpenFHE.
	std::vector<int> indices_rotacion {2, -1};
	cc->EvalRotateKeyGen(keys.secretKey, indices_rotacion);

	/// 3. Encriptar datos de entrada.
	lbcrypto::Plaintext a_texto_plano = cc->MakeCKKSPackedPlaintext(a);
	lbcrypto::Plaintext b_texto_plano = cc->MakeCKKSPackedPlaintext(b);
	lbcrypto::Ciphertext<lbcrypto::DCRTPoly> a_ct = cc->Encrypt(keys.publicKey, a_texto_plano);
	lbcrypto::Ciphertext<lbcrypto::DCRTPoly> b_ct = cc->Encrypt(keys.publicKey, b_texto_plano);

	// 3.1 Copiar ciphertext en OpenFHE.
	lbcrypto::Ciphertext<lbcrypto::DCRTPoly> c_ct(b_ct);

	/// 4. Obtener los parámetros de OpenFHE en bruto.
	auto raw_params = FIDESlib::CKKS::GetRawParams(cc);
	/// 5. Adaptar los parámetros de FIDESlib según los de OpenFHE.
	auto p =  params.adaptTo(raw_params);
	/// 6. Generar contexto de FIDESlib.
	FIDESlib::CKKS::Context gpu_cc (p, {0});

	/// 6.1 Carga de claves de multiplicación.
	/// 6.1.1 Extraer la clave del contexto usando la siguiente función que recibe las claves generadas de parámetro.
	auto clave_evaluacion = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);
	/// 6.1.2 Crear un objeto clave de gpu en el que alamcenaremos la clave anterior.
	FIDESlib::CKKS::KeySwitchingKey clave_evaluacion_gpu(gpu_cc);
	/// 6.1.3 Inicializar el objeto anterior con la clave extraida del contexto de OpenFHE.
	clave_evaluacion_gpu.Initialize(gpu_cc, clave_evaluacion);
	/// 6.1.4 Añadir la clave de evaluación al contexto de GPU para poder usarla después.
	gpu_cc.AddEvalKey(std::move(clave_evaluacion_gpu));
	/// 6.2 Carga de claves de rotación.
	/// 6.2.1 Iterar por todos los índices de rotación para cargar sus claves correspondientes.
	for (int i : indices_rotacion) {
		/// 6.2.2 Extraer la clave del contexto. Se da de parámetro las claves de encriptación, el índice deseado y el contexto de cpu.
		auto clave_rotacion = FIDESlib::CKKS::GetRotationKeySwitchKey(keys, i, cc);
		/// 6.2.3 Crear un objeto clave de gpu en el que almacenar la clave anterior.
		FIDESlib::CKKS::KeySwitchingKey clave_rotacion_gpu(gpu_cc);
		/// 6.2.4 Inicializar el objeto anterior con la clave extraida del contexto de OpenFHE.
		clave_rotacion_gpu.Initialize(gpu_cc, clave_rotacion);
		/// 6.2.5 Añadir la clave de rotación al contexto de GPU para poder usarla después.
		gpu_cc.AddRotationKey(i, std::move(clave_rotacion_gpu));
	}

	/// 7. Eliminar metadatos irrelevantes de textos cifrados.
	FIDESlib::CKKS::RawCipherText a_ct_raw = FIDESlib::CKKS::GetRawCipherText(cc, a_ct);
	FIDESlib::CKKS::RawCipherText b_ct_raw = FIDESlib::CKKS::GetRawCipherText(cc, b_ct);
	/// 8. Cargar en GPU los textos cifrados.
	FIDESlib::CKKS::Ciphertext a_ct_gpu (gpu_cc, a_ct_raw);
	FIDESlib::CKKS::Ciphertext b_ct_gpu (gpu_cc, b_ct_raw);

	/// 8.1 Copiar ciphertext en GPU.
	FIDESlib::CKKS::Ciphertext copia(gpu_cc);
	copia.copy(a_ct_gpu);

	/// 9. Operar.
	/// 9.1 Suma.	
	a_ct_gpu.add(b_ct_gpu);
	/// 9.2 Multiplicación. Se requiere la clave de multiplicación que obtendremos del contexto.;
	a_ct_gpu.mult(b_ct_gpu, gpu_cc.GetEvalKey());
	/// 9.2.1 Multiplicación alternativa. Si no se tiene el contexto de GPU disponible se puede obtener desde el propio ciphertext..
	// a_ct_gpu.mult(b_ct_gpu, b_ct_gpu.cc.GetEvalKey());
	/// 9.3 Rotación de elementos.
	int indice_rotacion = 2;
	a_ct_gpu.rotate(indice_rotacion, gpu_cc.GetRotationKey(indice_rotacion));
	/// 9.3.1 Rotación alternativa. Obtener contexto de GPU desde el ciphertext.
	// a_ct_gpu.rotate(i, a_ct_gpu.cc.GetRotationKey(i));
	/// 9.4 Cuadrado de un ciphertext
	a_ct_gpu.square(gpu_cc.GetEvalKey()); // a_ct_gpu.square(gpu_cc.GetEvalKey(), true);  

	/// 10. Volcar datos de vuelta.
	a_ct_gpu.store(gpu_cc, a_ct_raw);
	lbcrypto::Ciphertext<lbcrypto::DCRTPoly> res_openfhe(a_ct);
	FIDESlib::CKKS::GetOpenFHECipherText(res_openfhe, a_ct_raw);

	/// 11. Desencriptar.
	lbcrypto::Plaintext res_pt_fideslib;
	cc->Decrypt(keys.secretKey, res_openfhe, &res_pt_fideslib);
	res_pt_fideslib->SetLength(a.size());
	auto res_fideslib = res_pt_fideslib->GetCKKSPackedValue();

	for(size_t i = 0; i < a.size(); ++i) {
		a[i] = res_fideslib[i].real();
	}

	/// ATENCIÓN. Cuando no quieras usar más un contexto, usa estos métodos para ahorrar memoria.
	cc->ClearEvalMultKeys();
	cc->ClearEvalAutomorphismKeys();
}

void operaciones_openfhe_bootstrap(std::vector<double> &a, const std::vector<double> &b) {

	/// 1. Generar contexto.
	lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc = generate_context();

	/// 2. Crear claves.
	auto keys = cc->KeyGen();
	/// 2.1 Crear clave de multiplicación en OpenFHE.
	cc->EvalMultKeyGen(keys.secretKey);
	/// 2.2 Setup del bootstrapping.
	/// 2.2.1 Setup inicial. Se necesita pasar el presupuesto de niveles igual al usado al crear el contexto 
	/// y la cantidad de slots para la que se desea hacer bootstrapping.
    cc->EvalBootstrapSetup(level_budget, {0 ,0}, num_slots);
	/// 2.2.2 Generar las claves necesarias para el bootstrapping. Se necesita la clave privada y número de slots.
	cc->EvalBootstrapKeyGen(keys.secretKey, num_slots);
	/// 2.2.3 Generar las precomputaciones necesarias. Se necesita el número de slots.
	cc->EvalBootstrapPrecompute(num_slots);

	/// 3. Encriptar datos de entrada.
	lbcrypto::Plaintext a_texto_plano = cc->MakeCKKSPackedPlaintext(a);
	lbcrypto::Plaintext b_texto_plano = cc->MakeCKKSPackedPlaintext(b);
	lbcrypto::Ciphertext<lbcrypto::DCRTPoly> a_texto_cifrado = cc->Encrypt(keys.publicKey, a_texto_plano);
	lbcrypto::Ciphertext<lbcrypto::DCRTPoly> b_texto_cifrado = cc->Encrypt(keys.publicKey, b_texto_plano);

	/// 4. Operar para gastar niveles
	/// 4.2 Multiplicación.
	std::cout << "Niveles antes de multiplicar: " << a_texto_cifrado->GetLevel() << std::endl;
	auto res_texto_cifrado = cc->EvalMult(a_texto_cifrado, b_texto_cifrado);
	res_texto_cifrado = cc->EvalMult(res_texto_cifrado, b_texto_cifrado);
	res_texto_cifrado = cc->EvalMult(res_texto_cifrado, b_texto_cifrado);
	std::cout << "Niveles después de multiplicar: " << res_texto_cifrado->GetLevel() << std::endl;

	/// 5. Bootstraping 
	/// 5.1 Sin medición de tiempo.
	//cc->EvalBootstrap(res_texto_cifrado);
	/// 5.2 Con medición de tiempo.
	auto start = std::chrono::high_resolution_clock::now();
	cc->EvalBootstrap(res_texto_cifrado);
	auto end = std::chrono::high_resolution_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::cout << "Niveles después de bootstrap: " << res_texto_cifrado->GetLevel() << std::endl;
	std::cout << "Tiempo boostrap (ms): " << elapsed.count() << std::endl;
	/// 5.3 Estableciendo la cantidad de slots que te interesan.
	//a_texto_cifrado->SetSlots(1);
	//cc->EvalBootstrap(res_texto_cifrado);
	/// 5.3.1 Quizas despúes de hacer bootstrapping te interesa volver a aumentar la cantidad de slots.
	//a_texto_cifrado->SetSlots(num_slots);

	/// 6. Desencriptar texto cifrado.
	lbcrypto::Plaintext res_texto_plano;
	cc->Decrypt(keys.secretKey, res_texto_cifrado, &res_texto_plano);
	res_texto_plano->SetLength(a.size());
	auto res = res_texto_plano->GetCKKSPackedValue();

	for (int i = 0; i < a.size(); ++i) {
		a[i] = res[i].real();
	}

	/// ATENCIÓN. Cuando no quieras usar más un contexto, usa estos métodos para ahorrar memoria.
	cc->ClearEvalMultKeys();
	cc->ClearEvalAutomorphismKeys();
}

void operaciones_fideslib_bootstrap(std::vector<double> &a, const std::vector<double> &b) {
	/// 1. Generar contexto.
	lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc = generate_context();

	/// 2. Crear claves.
	auto keys = cc->KeyGen();
	/// 2.1 Crear clave de multiplicación en OpenFHE.
	cc->EvalMultKeyGen(keys.secretKey);
	/// 2.2 Setup del bootstrapping.
	/// 2.2.1 Setup inicial. Se necesita pasar el presupuesto de niveles igual al usado al crear el contexto 
	/// y la cantidad de slots para la que se desea hacer bootstrapping.
    cc->EvalBootstrapSetup(level_budget, {0 ,0}, num_slots);
	/// 2.2.2 Generar las claves necesarias para el bootstrapping. Se necesita la clave privada y número de slots.
	cc->EvalBootstrapKeyGen(keys.secretKey, num_slots);
	/// 2.2.3 Generar las precomputaciones necesarias. Se necesita el número de slots.
	cc->EvalBootstrapPrecompute(num_slots);

	/// 3. Encriptar datos de entrada.
	lbcrypto::Plaintext a_texto_plano = cc->MakeCKKSPackedPlaintext(a);
	lbcrypto::Plaintext b_texto_plano = cc->MakeCKKSPackedPlaintext(b);
	lbcrypto::Ciphertext<lbcrypto::DCRTPoly> a_ct = cc->Encrypt(keys.publicKey, a_texto_plano);
	lbcrypto::Ciphertext<lbcrypto::DCRTPoly> b_ct = cc->Encrypt(keys.publicKey, b_texto_plano);

	/// 4. Obtener los parámetros de OpenFHE en bruto.
	auto raw_params = FIDESlib::CKKS::GetRawParams(cc);
	/// 5. Adaptar los parámetros de FIDESlib según los de OpenFHE.
	auto p =  params.adaptTo(raw_params);
	/// 6. Generar contexto de FIDESlib.
	FIDESlib::CKKS::Context gpu_cc (p, {0});

	/// 6.1 Carga de claves de multiplicación.
	/// 6.1.1 Extraer la clave del contexto usando la siguiente función que recibe las claves generadas de parámetro.
	auto clave_evaluacion = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);
	/// 6.1.2 Crear un objeto clave de gpu en el que alamcenaremos la clave anterior.
	FIDESlib::CKKS::KeySwitchingKey clave_evaluacion_gpu(gpu_cc);
	/// 6.1.3 Inicializar el objeto anterior con la clave extraida del contexto de OpenFHE.
	clave_evaluacion_gpu.Initialize(gpu_cc, clave_evaluacion);
	/// 6.1.4 Añadir la clave de evaluación al contexto de GPU para poder usarla después.
	gpu_cc.AddEvalKey(std::move(clave_evaluacion_gpu));
	/// 6.2 Precomputaciones necesarias. Se necesitan tanto las claves como el número de slots sobre el que se trabajará.
	FIDESlib::CKKS::AddBootstrapPrecomputation(cc, keys, num_slots, gpu_cc);

	/// 7. Eliminar metadatos irrelevantes de textos cifrados.
	FIDESlib::CKKS::RawCipherText a_ct_raw = FIDESlib::CKKS::GetRawCipherText(cc, a_ct);
	FIDESlib::CKKS::RawCipherText b_ct_raw = FIDESlib::CKKS::GetRawCipherText(cc, b_ct);
	/// 8. Cargar en GPU los textos cifrados.
	FIDESlib::CKKS::Ciphertext a_ct_gpu (gpu_cc, a_ct_raw);
	FIDESlib::CKKS::Ciphertext b_ct_gpu (gpu_cc, b_ct_raw);

	/// 9. Operar.
	/// 9.1 Multiplicación. Se requiere la clave de multiplicación que obtendremos del contexto.;
	a_ct_gpu.mult(b_ct_gpu, gpu_cc.GetEvalKey());
	a_ct_gpu.mult(b_ct_gpu, gpu_cc.GetEvalKey());

	/// 10. Bootstrapping (es necesario especificar la cantidad de slots).
	FIDESlib::CKKS::Bootstrap(a_ct_gpu, num_slots);

	/// 11. Volcar datos de vuelta.
	a_ct_gpu.store(gpu_cc, a_ct_raw);
	lbcrypto::Ciphertext<lbcrypto::DCRTPoly> res_openfhe(a_ct);
	FIDESlib::CKKS::GetOpenFHECipherText(res_openfhe, a_ct_raw);

	/// 12. Desencriptar.
	lbcrypto::Plaintext res_pt_fideslib;
	cc->Decrypt(keys.secretKey, res_openfhe, &res_pt_fideslib);
	res_pt_fideslib->SetLength(a.size());
	auto res_fideslib = res_pt_fideslib->GetCKKSPackedValue();

	for(size_t i = 0; i < a.size(); ++i) {
		a[i] = res_fideslib[i].real();
	}

	/// ATENCIÓN. Cuando no quieras usar más un contexto, usa estos métodos para ahorrar memoria.
	cc->ClearEvalMultKeys();
	cc->ClearEvalAutomorphismKeys();
}

int main() {
	
	/// (Opcional según lo que se requiera) Cargar datos de un CSV.
	std::vector<std::vector<double>> csv;
	load_csv("./../data/vectorB.csv", csv);

	const std::vector<double> datos = {1,2,1,1,1,1,1,2};

	/// Declarar los vectores manualmente.
	std::vector<double> vectorA = datos;
	std::vector<double> vectorB = datos;

	operaciones_normal(vectorA, vectorB);

	for (auto a : vectorA) {
		std::cout << a << ", ";
	}
	std::cout << std::endl;

	vectorA = datos;
	operaciones_openfhe(vectorA, vectorB);

	for (auto a : vectorA) {
		std::cout << a << ", ";
	}
	std::cout << std::endl;

	vectorA = datos;
	operaciones_fideslib(vectorA, vectorB);

	for (auto a : vectorA) {
		std::cout << a << ", ";
	}
	std::cout << std::endl;

	operaciones_openfhe_bootstrap(vectorA, vectorB);
	for (auto a : vectorA) {
		std::cout << a << ", ";
	}
	std::cout << std::endl;

	operaciones_fideslib_bootstrap(vectorA, vectorB);
	for (auto a : vectorA) {
		std::cout << a << ", ";
	}
	std::cout << std::endl;
}
