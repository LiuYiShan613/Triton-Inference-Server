#include <iostream>
#include <string>

#include "grpc_client.h"
#include "http_client.h"
#include "model_config.pb.h"

int main() {
    std::string url = "localhost:8000";  
    std::string model_name = "yolov8n";

    // create HTTP client
    std::unique_ptr<triton::client::InferenceServerHttpClient> client;
    triton::client::InferenceServerHttpClient::Create(&client, url, false);

    // load model
    triton::client::Error err_lm  = client->LoadModel(model_name);
    if (err_lm.IsOk()) {
        std::cout << "model " << model_name << " load successfully" << std::endl;
    } else {
        std::cerr << "model load failed：" << err_lm.Message() << std::endl;
    }

    // check if model ready (after load)
    bool is_ready = false;
    triton::client::Error err_1 = client->IsModelReady(&is_ready, model_name);

    if (!err_1.IsOk()) {
        std::cerr << "Model status query failed: " << err_1.Message() << std::endl;
        return 1;
    }

    if (is_ready) {
        std::cout << "✅ model " << model_name << " is ready to do inference" << std::endl;
    } else {
        std::cout << "❌ model " << model_name << " not ready" << std::endl;
    }

    // unload model
    triton::client::Error err_um  = client->UnloadModel(model_name);
    if (err_um.IsOk()) {
        std::cout << "✅ model " << model_name << "  unload complete" << std::endl;
    } else {
        std::cerr << "❌ unload fall：" << err_um.Message() << std::endl;
    }

    // check if model unload
    bool is_ready_2 = false;
    triton::client::Error err_2 = client->IsModelReady(&is_ready_2, model_name);

    if (!err_2.IsOk()) {
        std::cerr << "Model status query failed: " << err_2.Message() << std::endl;
        return 1;
    }

    if (is_ready_2) {
        std::cout << "✅ model " << model_name << " is ready to do inference" << std::endl;
    } else {
        std::cout << "❌ model " << model_name << " is ready" << std::endl;
    }
    
    return 0;
}
