#include <iostream>
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/util/memmapped_file_system.h"
#include "tensorflow/core/util/memmapped_file_system.pb.h"
// #include "tensorflow/data_flow_ops.h"
// #include "tensorflow/core/kernels/fifo_queue.h"
// #include "tensorflow/core/platform/types.h"

#include "tensorflow/cc/framework/scope.h"
//#include "tensorflow/cc/ops/standard_ops.h"
//#include "tensorflow/cc/training/coordinator.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/error_codes.pb.h"
#include "tensorflow/core/lib/core/notification.h"
// #include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/env.h"
// #include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/queue_runner.pb.h"
#include "tensorflow/core/public/session.h"
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <map>
#include <chrono> 
#include <iomanip>
#include <memory>

extern "C" {
#include "lpcnet_interface.h"
}

#define CREATE_F32_FILE false
static clock_t start, end, mid; 
//static tensorflow::Tensor input_lengths_t(tensorflow::DT_INT32, tensorflow::TensorShape({1}));
//static auto flat_input_lengths = input_lengths_t.flat<int>();
static std::vector<std::pair<std::string, tensorflow::Tensor>> inputs;
static tensorflow::Session *session;

static std::map<char, int> char2seq =  { 
{'_', 0},{'~', 1},{'A', 2},{'B', 3},{'C', 4},{'D', 5},{'E', 6},{'F', 7},{'G', 8},{'H', 9},{'I', 10},{'J', 11},{'K', 12},{'L', 13},{'M', 14},{'N', 15},{'O', 16},{'P', 17},{'Q', 18},{'R', 19},{'S', 20},{'T', 21},{'U', 22},{'V', 23},{'W', 24},{'X', 25},{'Y', 26},{'Z', 27},{'a', 28},{'b', 29},{'c', 30},{'d', 31},{'e', 32},{'f', 33},{'g', 34},{'h', 35},{'i', 36},{'j', 37},{'k', 38},{'l', 39},{'m', 40},{'n', 41},{'o', 42},{'p', 43},{'q', 44},{'r', 45},{'s', 46},{'t', 47},{'u', 48},{'v', 49},{'w', 50},{'x', 51},{'y', 52},{'z', 53},{'!', 54},{'\'',55},{'(', 56},{')', 57},{',', 58},{'-', 59},{'.', 60},{':', 61},{';', 62},{'?', 63},{' ', 64}
};



int init_lpctron(char *argv) {
    tensorflow::GraphDef graph_def;
    std::cout << "Loading model..." << std::endl;


    tensorflow::Status status;
    tensorflow::MemmappedEnv* mmap_env_;
    tensorflow::SessionOptions options;


    mmap_env_ = new tensorflow::MemmappedEnv(tensorflow::Env::Default());

    bool is_mmap = std::string(argv).find(".pbmm") != std::string::npos;
    if (!is_mmap) {
        std::cerr << "Warning: reading entire model file into memory. Transform model file into an mmapped graph to reduce heap usage." << std::endl;
    } else {

        /* Procedure to convert protobuf model to memory mapped model 
        bazel build tensorflow/contrib/util:convert_graphdef_memmapped_format
        bazel-bin/tensorflow/contrib/util/convert_graphdef_memmapped_format --in_graph=inference_model.pb --out_graph=inference_model.pbmm
        */

        status = mmap_env_->InitializeFromFile(argv);
        if (!status.ok()) {
            std::cerr << status << std::endl;
            return -1;
        }

        options.config.mutable_graph_options()
            ->mutable_optimizer_options()
            ->set_opt_level(tensorflow::OptimizerOptions::L0);
        options.env = mmap_env_;
    }

    
    if (!status.ok())
    {
        std::cout << status.ToString() << std::endl;
        return -1;
    }
    std::cout << "model loaded" << std::endl;

    std::cout << "Creating session" << std::endl;

    tensorflow::ConfigProto & config = options.config;
    config.set_inter_op_parallelism_threads(1);
    config.set_intra_op_parallelism_threads(1);
    config.set_use_per_session_threads(true);

    status = tensorflow::NewSession(options, &session);


    if (is_mmap) {
        status = ReadBinaryProto(mmap_env_,
                tensorflow::MemmappedFileSystem::kMemmappedPackageDefaultGraphDef,
                &graph_def);

        std::cout << "Mmaping the model file" << std::endl;
    } else {
        status = tensorflow::ReadBinaryProto(tensorflow::Env::Default(), argv, &graph_def);
        std::cout << "Loading PB Memory " << std::endl;
    }


    if (!status.ok())
    {
        std::cout << status.ToString() << "\n";
        return -1;
    }
    std::cout << "Session created" << std::endl;

    std::cout << "Connecting session to graph..." << std::endl;
    status = session->Create(graph_def);
    if (!status.ok()) {
        std::cout << "Create error: " << status.ToString() << "\n";
        return -1;
    }
    std::cout << "Graph connected" << std::endl;
    init_lpcnet();
    return 0;
}

int tts_lpctron(const std::string& text, void (*pcm_callback)(short *pcm, int pcm_size)) {
    std::vector<int> seq_inputs;
	start = std::clock();         

	static tensorflow::Tensor input_lengths_t(tensorflow::DT_INT32, tensorflow::TensorShape({1}));
	static auto flat_input_lengths = input_lengths_t.flat<int>(); 
    for(const char& c : text) {
        if (char2seq.find(c) != char2seq.end()) {
            seq_inputs.push_back(char2seq.at(c));
            std::cout << c << ":" << char2seq.at(c) << " ";
        }
    }
    std::cout << std::endl;

    int inputs_size = seq_inputs.size();
    if (inputs_size > 0) {

        tensorflow::Tensor inputs_t(tensorflow::DT_INT32, tensorflow::TensorShape({1, inputs_size}));

        auto flat_inputs = inputs_t.tensor<int, 2>();
        flat_input_lengths(0) = inputs_size;
        for (int i = 0; i < inputs_size; i++) {
            flat_inputs(0, i) = seq_inputs[i];
        }

        inputs.clear();
        inputs.emplace_back(std::string("inputs"), inputs_t);
        inputs.emplace_back(std::string("input_lengths"), input_lengths_t);

        std::cout << "Input created" << std::endl;

        std::vector<tensorflow::Tensor> outputs;


        std::cout << "Running inference..." << std::endl;

        tensorflow::Status status = session->Run(inputs, {std::string("model/inference/add")}, {}, &outputs);
        if (!status.ok()) {
            std::cout << status.ToString() << "\n";
            return 1;
        }
        std::cout << "got " << (int)outputs.size() << " outputs" << std::endl;
        if (outputs.size() == 1) {
            const tensorflow::Tensor& mels = outputs[0];
            auto mels_map = outputs[0].tensor<float, 3>();
            int dim0 = mels.shape().dim_size(0);
            int dim1 = mels.shape().dim_size(1);
            int dim2 = mels.shape().dim_size(2);
            std::cout << "mels dimensions = " << dim0 << ", " << dim1 << ", " << dim2 << std::endl;
            // float *mels_data = new float[dim0 * dim1 * dim2];
            std::vector<float> mels_data;

            for (int i = 0; i < dim0; i++) {
                for (int j = 0; j < dim1; j++) {
                    for (int k = 0; k < dim2; k++) {
                        // *(mels_data + i * dim1 * dim2 + j * dim2 + k) = mels_map(i, j, k);
                        mels_data.push_back(mels_map(i, j, k));
                    }
                }
            }
            
#if CREATE_F32_FILE
            std::ofstream data_file;      // pay attention here! ofstream
            data_file.open("inference_model_cpp.f32", std::ios::out | std::ios::binary);
            data_file.write(reinterpret_cast<char*>(&mels_data[0]), mels_data.size()*sizeof(float)); 
            data_file.close();
#endif        
            /* 
            float *features = new float[mels_data.size()];
            for (int i = 0; i < mels_data.size(); i++) {
                features[i] = mels_data[i];
            }
            */
			mid  = clock();
			double time_mid = double(mid - start) / double(CLOCKS_PER_SEC);
			std::cout << "Time taken by program Tactron2 : " << std::fixed
					<< time_mid << std::setprecision(5);
			std::cout << " sec " << std::endl;
			
            std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
            run_lpcnet(reinterpret_cast<float*>(&mels_data[0]), mels_data.size(), pcm_callback);
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            std::cout << "LPCnet took- "
                    << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
                    << "us.\n";

            //delete[] features;
            return 0;
            // std::cout << "Writing " << mels_data.size() << " values to file..." << std::endl;
            // // std::ofstream outfile ("inference_model_cpp.f32",std::ofstream::binary);
            // // outfile.write((char *)mels_data,(dim0 * dim1 * dim2)*sizeof(float));
            // // outfile.close();
            // std::ofstream data_file;      // pay attention here! ofstream
            // data_file.open("inference_model_cpp.f32", std::ios::out | std::ios::binary);
            // data_file.write(reinterpret_cast<char*>(&mels_data[0]), mels_data.size()*sizeof(float)); 
            // data_file.close();

            // std::cout << "File written" << std::endl;

            // delete[] mels_data;
        }

    }

    return -1;
}
#define USE_C_STYLE_FILE_WRITE false

#if USE_C_STYLE_FILE_WRITE
FILE *pcm_file;
#else
static std::ofstream pcm_file;
#endif

void callback(short *pcm, int num_elements) {
    // for (int i = 0; i < num_elements; i++) {
    //     if (pcm[i] != 0) {
    //         std::cout << "non zero " << pcm[i] << std::endl;
    //     }
    // }
#if USE_C_STYLE_FILE_WRITE
    fwrite(pcm, sizeof(short), num_elements, pcm_file);
#else
    // pcm_file.write(pcm, num_elements); 
    // pcm_file << pcm;
    pcm_file.write((char *)pcm, num_elements * sizeof(short));
#endif
}

int main(int argc, char *argv[])
{
    #if true

	if(argc < 2) {
			std::cout << "usage program <model> <text_file>"<<std::endl;
			return 1;
	}
    if (init_lpctron(argv[1]) == 0) {
        
        #if USE_C_STYLE_FILE_WRITE
        pcm_file = fopen("output.pcm", "wb");
        #else
        pcm_file.open("output.pcm", std::ios::binary);
        #endif

        std::ifstream textfile (argv[2]);
        std::string str((std::istreambuf_iterator<char>(textfile)),
                    std::istreambuf_iterator<char>());
        if (tts_lpctron(str, &callback) != 0) {
            std::cout << "error with tts_lpctron" << std::endl;
        }
        #if USE_C_STYLE_FILE_WRITE
        fclose(pcm_file);
        #else
        pcm_file.close();
        #endif
    } else {
        std::cout << "error with init_lpctron" << std::endl;
    }
    std::cout << "done" << std::endl;
    end  = clock();
    double time_taken = double(end - start) / double(CLOCKS_PER_SEC); 
    std::cout << "Time taken by program is : " << std::fixed  
         << time_taken << std::setprecision(5); 
    std::cout << " sec " << std::endl; 
    return 0; 
    #endif

    #if 0 
    init_lpcnet();
    pcm_file.open("output.pcm", std::ios::out | std::ios::binary);

    std::ifstream file("inference_model_cpp.f32", std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<float> buffer(size / sizeof(float));
    if (file.read((char *)buffer.data(), size))
    {
        std::cout << "running lpcnet..." << std::endl;
        run_lpcnet(reinterpret_cast<float*>(&buffer[0]), buffer.size(), &callback);
        pcm_file.close();
    } else {
        std::cerr << "error reading f32 file" << std::endl;
    }

    #endif
	// while (textfile >> std::noskipws >> ch) {
	// 		//cout << ch;
	// 		it = char2seq.find(ch);
	// 		if(it != char2seq.end()) 
	// 			seq_inputs.push_back(it->second);

	// }


    // //std::vector<int> seq_inputs = {35, 32, 52, 64, 49, 36, 46, 47, 32, 42, 41, 64, 35, 42, 50, 64, 36, 46, 64, 47, 35, 32, 64, 50, 32, 28, 47, 35, 32, 45, 64, 47, 42, 31, 28, 52, 60, 1};
    // int inputs_size = seq_inputs.size();

    // tensorflow::Tensor inputs_t(tensorflow::DT_INT32, tensorflow::TensorShape({1, inputs_size}));
    // tensorflow::Tensor input_lengths_t(tensorflow::DT_INT32, tensorflow::TensorShape({1}));

    

    // auto flat_inputs = inputs_t.tensor<int, 2>();
    // auto flat_input_lengths = input_lengths_t.flat<int>();
    // flat_input_lengths(0) = inputs_size;
    // for (int i = 0; i < inputs_size; i++) {
    //     flat_inputs(0, i) = seq_inputs[i];
    // }

    // std::vector<std::pair<std::string, tensorflow::Tensor>> inputs;

    // inputs.emplace_back(std::string("inputs"), inputs_t);
    // inputs.emplace_back(std::string("input_lengths"), input_lengths_t);

    // std::cout << "Input created" << std::endl;

    // std::vector<tensorflow::Tensor> outputs;

    // std::cout << "Creating session" << std::endl;
    // tensorflow::Session *session;
    // status = tensorflow::NewSession(tensorflow::SessionOptions(), &session);
    // if (!status.ok())
    // {
    //     std::cout << status.ToString() << "\n";
    //     return 1;
    // }
    // std::cout << "Session created" << std::endl;

    // std::cout << "Connecting session to graph..." << std::endl;
    // status = session->Create(graph_def);
    // if (!status.ok()) {
    //     std::cout << "Create error: " << status.ToString() << "\n";
    //     return 1;
    // }
    // std::cout << "Graph connected" << std::endl;

    // std::cout << "Running inference..." << std::endl;

    // status = session->Run(inputs, {std::string("model/inference/add")}, {}, &outputs);
    // if (!status.ok()) {
    //     std::cout << status.ToString() << "\n";
    //     return 1;
    // }
    // std::cout << "got " << (int)outputs.size() << " outputs" << std::endl;
    // if (outputs.size() == 1) {
    //     const tensorflow::Tensor& mels = outputs[0];
    //     auto mels_map = outputs[0].tensor<float, 3>();
    //     int dim0 = mels.shape().dim_size(0);
    //     int dim1 = mels.shape().dim_size(1);
    //     int dim2 = mels.shape().dim_size(2);
    //     std::cout << "mels dimensions = " << dim0 << ", " << dim1 << ", " << dim2 << std::endl;
    //     // float *mels_data = new float[dim0 * dim1 * dim2];
    //     std::vector<float> mels_data;

    //     for (int i = 0; i < dim0; i++) {
    //         for (int j = 0; j < dim1; j++) {
    //             for (int k = 0; k < dim2; k++) {
    //                 // *(mels_data + i * dim1 * dim2 + j * dim2 + k) = mels_map(i, j, k);
    //                 mels_data.push_back(mels_map(i, j, k));
    //             }
    //         }
    //     }

    //     std::cout << "Writing " << mels_data.size() << " values to file..." << std::endl;
    //     // std::ofstream outfile ("inference_model_cpp.f32",std::ofstream::binary);
    //     // outfile.write((char *)mels_data,(dim0 * dim1 * dim2)*sizeof(float));
    //     // outfile.close();
    //     std::ofstream data_file;      // pay attention here! ofstream
    //     data_file.open("inference_model_cpp.f32", std::ios::out | std::ios::binary);
    //     data_file.write(reinterpret_cast<char*>(&mels_data[0]), mels_data.size()*sizeof(float)); 
    //     data_file.close();

    //     std::cout << "File written" << std::endl;

    //     // delete[] mels_data;
    // }


    return 0;
}
