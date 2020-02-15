#include<cufhe_gpu.cuh>
#include<tfhe++.hpp>
#include<memory>
#include<chrono>

using namespace TFHEpp;
using namespace cufhe;
using namespace std::chrono;

std::shared_ptr<cufhe::PubKey> tfhepp2cufhe(const TFHEpp::GateKey& src)
{
    auto pubkey = std::make_shared<cufhe::PubKey>();
    // FIXME: Check if TFHEpp's parameters are the same as cuFHE's.
    auto cufheParams = cufhe::GetDefaultParam();
    const int32_t n = cufheParams->lwe_n_;
    const int32_t N = cufheParams->tlwe_n_;
    const int32_t k = cufheParams->tlwe_k_;
    const int32_t l = cufheParams->tgsw_decomp_size_;
    const int32_t ksk_t = cufheParams->keyswitching_decomp_size_;
    const int32_t ksk_n = N * k;
    const int32_t ksk_base = (1 << cufheParams->keyswitching_decomp_bits_);
    // Read the bootstrapping key.
    for (int p = 0; p < n; p++) {
        const TFHEpp::TRGSWFFTlvl1& trgswfft = src.bkfftlvl01[p];
        for (int q = 0; q < (k + 1) * l; q++) {
            for (int r = 0; r < (k + 1); r++) {
                TFHEpp::Polynomiallvl1 poly;
                TFHEpp::TwistFFTlvl1(poly, trgswfft[q][r]);
                for (int s = 0; s < N; s++) {
                    int index = ((p * ((k + 1) * l) + q) * (k + 1) + r) * N + s;
                    pubkey->bk_->data()[index] = poly[s];
                }
            }
        }
    }
    // Read the key switch key.
    for (int p = 0; p < ksk_n; p++) {
        for (int q = 0; q < ksk_t; q++) {
            // r = 0
            {
                cufhe::LWESample to = pubkey->ksk_->ExtractLWESample(
                    pubkey->ksk_->GetLWESampleIndex(p, q, 0));
                for (int s = 0; s < n; s++)
                    to.data()[s] = 0;
                to.data()[n] = 0;
            }
            // r >= 1
            for (int r = 1; r < ksk_base; r++) {
                assert(static_cast<size_t>(p) < src.ksk.size());
                assert(static_cast<size_t>(q) < src.ksk[p].size());
                assert(static_cast<size_t>(r - 1) < src.ksk[p][q].size());
                const TFHEpp::TLWElvl0& from = src.ksk[p][q][r - 1];
                cufhe::LWESample to = pubkey->ksk_->ExtractLWESample(
                    pubkey->ksk_->GetLWESampleIndex(p, q, r));
                for (int s = 0; s < n; s++) {
                    assert(static_cast<size_t>(s) < from.size());
                    to.data()[s] = from[s];
                }
                to.data()[n] = from[n];
            }
        }
    }
    return pubkey;
}

int main(){
    SecretKey sk;
    GateKey* gk = new GateKey(sk);

    cudaSetDevice(0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    SetSeed(); // set random seed
    uint32_t kNumSMs = prop.multiProcessorCount;

    std::shared_ptr<cufhe::PubKey> pub_key = tfhepp2cufhe(*gk);
    cuFHETRLWElvl1 cutrlwe;
    Ctxt ct;
    Stream st;
    st.Create();
    Initialize(*pub_key);
    Synchronize();

    random_device seed_gen;
    default_random_engine engine(seed_gen());
    uniform_int_distribution<uint32_t> binary(0, 1);
    cout << "lvl1" << endl;
    constexpr uint32_t num_test = 1000;
    for (int test = 0; test < num_test; test++) {
        bool p = binary(engine)>0;
        array<uint32_t, DEF_N> pmu;
        pmu[0] = p ? DEF_μ : -DEF_μ;
        for (int i = 1; i < DEF_N; i++) pmu[i] = 0;
        TRLWElvl1 trlwe = trlweSymEncryptlvl1(pmu,DEF_αbk,sk.key.lvl1);
        cutrlwe.trlwehost = trlwe;
        high_resolution_clock::time_point begin = high_resolution_clock::now();
        SampleExtractAndKeySwitch(ct,cutrlwe,st);
        high_resolution_clock::time_point end = high_resolution_clock::now();
        Synchronize();
        TLWElvl0 tlwe;
        for(int i = 0;i<=DEF_n;i++) tlwe[i] = ct.lwe_sample_->data()[i];
        uint8_t p2 = tlweSymDecryptlvl0(tlwe, sk.key.lvl0);
        assert(static_cast<int>(p) == static_cast<int>(p2));
	//std::cerr << duration_cast<microseconds>(end - begin).count() << std::endl;
    }
    cout << "Passed" << endl;
}