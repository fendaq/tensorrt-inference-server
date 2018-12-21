// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <string>
#include "cuda/include/cuda.h"
#include "src/core/logging.h"
#include "src/core/model_config.h"
#include "src/core/model_config.pb.h"
#include "src/core/model_config_cuda.h"
#include "src/custom/addsub/kernel.h"
#include "src/servables/custom/custom.h"

// This custom backend takes two INT32 input tensors (any shape but
// must have the same shape) and produces two output tensors (with
// same shape as the inputs). The input tensors must be named "INPUT0"
// and "INPUT1". The output tensors must be named "OUTPUT0" and
// "OUTPUT1". This backend does element-wise operation to produce:
//
//   OUTPUT0 = INPUT0 + INPUT1
//   OUTPUT1 = INPUT0 - INPUT1
//

namespace nvidia { namespace inferenceserver { namespace custom {
namespace addsub {

// Integer error codes. TRTIS requires that success must be 0. All
// other codes are interpreted by TRTIS as failures.
enum ErrorCodes {
  kSuccess,
  kUnknown,
  kInvalidModelConfig,
  kGpuNotSupported,
  kInputOutputShape,
  kInputName,
  kOutputName,
  kInputOutputDataType,
  kInputContents,
  kInputSize,
  kOutputBuffer,
  kCudaMalloc,
  kCudaMemcpy
};

// Context object. All state must be kept in this object.
class Context {
 public:
  Context(const ModelConfig& config, const int gpu_device);
  ~Context();

  // Initialize the context. Validate that the model configuration,
  // etc. is something that we can handle.
  int Init();

  // Perform custom execution on the payloads.
  int Execute(
      const uint32_t payload_cnt, CustomPayload* payloads,
      CustomGetNextInputFn_t input_fn, CustomGetOutputFn_t output_fn);

 private:
  int GetInputTensorCPU(
      CustomGetNextInputFn_t input_fn, void* input_context, const char* name,
      const size_t expected_byte_size, std::vector<int32_t>* input);
  int GetInputTensorGPU(
      CustomGetNextInputFn_t input_fn, void* input_context, const char* name,
      const size_t expected_byte_size, int32_t* input);

  int ExecuteCPU(
      const uint32_t payload_cnt, CustomPayload* payloads,
      CustomGetNextInputFn_t input_fn, CustomGetOutputFn_t output_fn);
  int ExecuteGPU(
      const uint32_t payload_cnt, CustomPayload* payloads,
      CustomGetNextInputFn_t input_fn, CustomGetOutputFn_t output_fn);

  // The model configuration.
  const ModelConfig model_config_;

  // The GPU device ID to execute on or CUSTOM_NO_GPU_DEVICE if should
  // execute on CPU.
  const int gpu_device_;

  // The size, in bytes, of batch-size 1 input or output tensor. Input
  // and output tensors are the same shape and data-type so they all
  // have the same size. To get the full size of an input/output need
  // to multiply this value by the batch-size.
  uint64_t batch1_byte_size_;

  // CUDA memory buffers for input and output tensors.
  size_t cuda_buffer_byte_size_;
  int32_t* cuda_input0_;
  int32_t* cuda_input1_;
  int32_t* cuda_output_;

  // The contexts executing on a GPU, the CUDA stream to use for the
  // execution.
  cudaStream_t stream_;
};

Context::Context(const ModelConfig& model_config, const int gpu_device)
    : model_config_(model_config), gpu_device_(gpu_device),
      batch1_byte_size_(0), cuda_buffer_byte_size_(0), cuda_input0_(nullptr),
      cuda_input1_(nullptr), cuda_output_(nullptr), stream_(nullptr)
{
}

Context::~Context()
{
  if (cuda_input0_ != nullptr) {
    cudaError_t cuerr = cudaFree(cuda_input0_);
    if (cuerr != cudaSuccess) {
      LOG_ERROR << "Failed to free cuda memory: " << cudaGetErrorString(cuerr);
    }
  }
  if (cuda_input1_ != nullptr) {
    cudaError_t cuerr = cudaFree(cuda_input1_);
    if (cuerr != cudaSuccess) {
      LOG_ERROR << "Failed to free cuda memory: " << cudaGetErrorString(cuerr);
    }
  }
  if (cuda_output_ != nullptr) {
    cudaError_t cuerr = cudaFree(cuda_output_);
    if (cuerr != cudaSuccess) {
      LOG_ERROR << "Failed to free cuda memory: " << cudaGetErrorString(cuerr);
    }
  }

  if (stream_ != nullptr) {
    cudaError_t cuerr = cudaStreamDestroy(stream_);
    if (cuerr != cudaSuccess) {
      LOG_ERROR << "Failed to destroy cuda stream: "
                << cudaGetErrorString(cuerr);
    }
    stream_ = nullptr;
  }
}

int
Context::Init()
{
  // There must be two inputs that have the same shape. The shape can
  // be anything since we are just going to do an element-wise add and
  // an element-wise subtract. The input data-type must be INT32. The
  // inputs must be named INPUT0 and INPUT1.
  if (model_config_.input_size() != 2) {
    return kInputOutputShape;
  }
  if (!CompareDims(
          model_config_.input(0).dims(), model_config_.input(1).dims())) {
    return kInputOutputShape;
  }
  if ((model_config_.input(0).data_type() != DataType::TYPE_INT32) ||
      (model_config_.input(1).data_type() != DataType::TYPE_INT32)) {
    return kInputOutputDataType;
  }
  if ((model_config_.input(0).name() != "INPUT0") ||
      (model_config_.input(1).name() != "INPUT1")) {
    return kInputName;
  }

  // There must be two outputs that have the same shape as the
  // inputs. The output data-type must be INT32. The outputs must be
  // named OUTPUT0 and OUTPUT1.
  if (model_config_.output_size() != 2) {
    return kInputOutputShape;
  }
  if (!CompareDims(
          model_config_.output(0).dims(), model_config_.output(1).dims()) ||
      !CompareDims(
          model_config_.output(0).dims(), model_config_.input(0).dims())) {
    return kInputOutputShape;
  }
  if ((model_config_.output(0).data_type() != DataType::TYPE_INT32) ||
      (model_config_.output(1).data_type() != DataType::TYPE_INT32)) {
    return kInputOutputDataType;
  }
  if ((model_config_.output(0).name() != "OUTPUT0") ||
      (model_config_.output(1).name() != "OUTPUT1")) {
    return kOutputName;
  }

  // Due to the above contraints, each input and output tensor will be
  // the same size (in bytes). Calculate that batch-1 size as it is
  // needed when reading and writing the tensors during execution.
  batch1_byte_size_ = GetByteSize(model_config_.input(0));

  // Additional initialization if executing on the GPU...
  if (gpu_device_ != CUSTOM_NO_GPU_DEVICE) {
    cudaError_t cuerr;
    // Allocate GPU memory buffers large enough for each input and
    // output. For performance we allocate once during initialization
    // instead of doing it each time we execute.
    cuda_buffer_byte_size_ =
        batch1_byte_size_ * std::max(1, model_config_.max_batch_size());
    cuerr = cudaMalloc(&cuda_input0_, cuda_buffer_byte_size_);
    if (cuerr != cudaSuccess) {
      LOG_ERROR << "unable to allocate memory for addsub: "
                << cudaGetErrorString(cuerr);
      return kCudaMalloc;
    }
    cuerr = cudaMalloc(&cuda_input1_, cuda_buffer_byte_size_);
    if (cuerr != cudaSuccess) {
      LOG_ERROR << "unable to allocate memory for addsub: "
                << cudaGetErrorString(cuerr);
      return kCudaMalloc;
    }
    cuerr = cudaMalloc(&cuda_output_, cuda_buffer_byte_size_);
    if (cuerr != cudaSuccess) {
      LOG_ERROR << "unable to allocate memory for addsub: "
                << cudaGetErrorString(cuerr);
      return kCudaMalloc;
    }

    // Create a CUDA stream for this context so that it executes
    // independently of other instances of this backend.
    const int cuda_stream_priority =
        GetCudaStreamPriority(model_config_.optimization().priority());
    cuerr = cudaStreamCreateWithPriority(
        &stream_, cudaStreamDefault, cuda_stream_priority);
    if (cuerr != cudaSuccess) {
      LOG_ERROR << "unable to create stream for addsub backend: "
                << cudaGetErrorString(cuerr);
    }
  }

  return kSuccess;
}

int
Context::GetInputTensorCPU(
    CustomGetNextInputFn_t input_fn, void* input_context, const char* name,
    const size_t expected_byte_size, std::vector<int32_t>* input)
{
  // The values for an input tensor are not necessarily in one
  // contiguous chunk, so we copy the chunks into 'input' vector. A
  // more performant solution would attempt to use the input tensors
  // in-place instead of having this copy.
  uint64_t total_content_byte_size = 0;

  while (true) {
    const void* content;
    uint64_t content_byte_size;
    if (!input_fn(input_context, name, &content, &content_byte_size)) {
      return kInputContents;
    }

    // If 'content' returns nullptr we have all the input.
    if (content == nullptr) {
      break;
    }

    // If the total amount of content received exceeds what we expect
    // then something is wrong.
    total_content_byte_size += content_byte_size;
    if (total_content_byte_size > expected_byte_size) {
      return kInputSize;
    }

    size_t content_elements = content_byte_size / sizeof(int32_t);
    input->insert(
        input->end(), static_cast<const int32_t*>(content),
        static_cast<const int32_t*>(content) + content_elements);
  }

  // Make sure we end up with exactly the amount of input we expect.
  if (total_content_byte_size != expected_byte_size) {
    return kInputSize;
  }

  return kSuccess;
}

int
Context::ExecuteCPU(
    const uint32_t payload_cnt, CustomPayload* payloads,
    CustomGetNextInputFn_t input_fn, CustomGetOutputFn_t output_fn)
{
  // Each payload represents a related set of inputs and required
  // outputs. Each payload may have a different batch size. The total
  // batch-size of all payloads will not exceed the max-batch-size
  // specified in the model configuration.

  // For performance, we would typically execute all payloads together
  // as a single batch by first gathering the inputs from across the
  // payloads and then scattering the outputs across the payloads.
  // Here, for simplicity and clarity, we instead process each payload
  // separately.
  int err;

  for (uint32_t pidx = 0; pidx < payload_cnt; ++pidx) {
    CustomPayload& payload = payloads[pidx];

    // For this payload the expected size of the input and output
    // tensors is determined by the batch-size of this payload.
    size_t batchn_byte_size = payload.batch_size * batch1_byte_size_;

    // Get the input tensors.
    std::vector<int32_t> input0;
    err = GetInputTensorCPU(
        input_fn, payload.input_context, "INPUT0", batchn_byte_size, &input0);
    if (err != kSuccess) {
      payload.error_code = err;
      continue;
    }

    std::vector<int32_t> input1;
    err = GetInputTensorCPU(
        input_fn, payload.input_context, "INPUT1", batchn_byte_size, &input1);
    if (err != kSuccess) {
      payload.error_code = err;
      continue;
    }

    // For each requested output get the buffer to hold the output
    // values and calculate the sum/difference directly into that
    // buffer.
    for (uint32_t oidx = 0; oidx < payload.output_cnt; ++oidx) {
      const char* output_name = payload.required_output_names[oidx];

      void* obuffer;
      if (!output_fn(
              payload.output_context, output_name, batchn_byte_size,
              &obuffer)) {
        payload.error_code = kOutputBuffer;
        break;
      }

      int32_t* output = static_cast<int32_t*>(obuffer);

      if (!strncmp(output_name, "OUTPUT0", strlen("OUTPUT0"))) {
        for (uint32_t i = 0; i < input0.size(); ++i) {
          output[i] = input0[i] + input1[i];
        }
      } else {
        for (uint32_t i = 0; i < input0.size(); ++i) {
          output[i] = input0[i] - input1[i];
        }
      }
    }
  }

  return kSuccess;
}

int
Context::GetInputTensorGPU(
    CustomGetNextInputFn_t input_fn, void* input_context, const char* name,
    const size_t expected_byte_size, int32_t* input)
{
  // The values for an input tensor are not necessarily in one
  // contiguous chunk, so we copy the chunks into 'input', which
  // points to CUDA memory.
  uint64_t total_content_byte_size = 0;

  while (true) {
    const void* content;
    uint64_t content_byte_size;
    if (!input_fn(input_context, name, &content, &content_byte_size)) {
      return kInputContents;
    }

    // If 'content' returns nullptr we have all the input.
    if (content == nullptr) {
      break;
    }

    // If the total amount of content received exceeds what we expect
    // then something is wrong.
    if ((total_content_byte_size + content_byte_size) > expected_byte_size) {
      return kInputSize;
    }

    cudaError_t cuerr = cudaMemcpyAsync(
        reinterpret_cast<char*>(input) + total_content_byte_size, content,
        content_byte_size, cudaMemcpyHostToDevice, stream_);
    if (cuerr != cudaSuccess) {
      LOG_ERROR << "failed to copy input values to GPU for addsub: "
                << cudaGetErrorString(cuerr);
      return kCudaMemcpy;
    }

    total_content_byte_size += content_byte_size;
  }

  // Make sure we end up with exactly the amount of input we expect.
  if (total_content_byte_size != expected_byte_size) {
    return kInputSize;
  }

  return kSuccess;
}

int
Context::ExecuteGPU(
    const uint32_t payload_cnt, CustomPayload* payloads,
    CustomGetNextInputFn_t input_fn, CustomGetOutputFn_t output_fn)
{
  // Each payload represents a related set of inputs and required
  // outputs. Each payload may have a different batch size. The total
  // batch-size of all payloads will not exceed the max-batch-size
  // specified in the model configuration.

  // For performance, we would typically execute all payloads together
  // as a single batch by first gathering the inputs from across the
  // payloads and then scattering the outputs across the payloads.
  // Here, for simplicity and clarity, we instead process each payload
  // separately.
  int err;

  cudaSetDevice(gpu_device_);

  for (uint32_t pidx = 0; pidx < payload_cnt; ++pidx) {
    CustomPayload& payload = payloads[pidx];

    // For this payload the expected size of the input and output
    // tensors is determined by the batch-size of this payload.
    size_t batchn_byte_size = payload.batch_size * batch1_byte_size_;
    size_t batchn_elements = payload.batch_size * batch1_byte_size_ /
                             GetDataTypeByteSize(DataType::TYPE_INT32);
    if (batchn_byte_size > cuda_buffer_byte_size_) {
      payload.error_code = kInputSize;
      continue;
    }

    // Copy the input tensors into the appropriate CUDA memory buffer.
    err = GetInputTensorGPU(
        input_fn, payload.input_context, "INPUT0", batchn_byte_size,
        cuda_input0_);
    if (err != kSuccess) {
      payload.error_code = err;
      continue;
    }

    err = GetInputTensorGPU(
        input_fn, payload.input_context, "INPUT1", batchn_byte_size,
        cuda_input1_);
    if (err != kSuccess) {
      payload.error_code = err;
      continue;
    }

    // For each requested output calculate the sum/difference directly
    // into the CUDA output buffer and then copy out.
    for (uint32_t oidx = 0; oidx < payload.output_cnt; ++oidx) {
      const char* output_name = payload.required_output_names[oidx];

      void* obuffer;
      if (!output_fn(
              payload.output_context, output_name, batchn_byte_size,
              &obuffer)) {
        payload.error_code = kOutputBuffer;
        break;
      }

      int block_size = 1024;
      int grid_size = (batchn_elements + block_size - 1) / block_size;

      if (!strncmp(output_name, "OUTPUT0", strlen("OUTPUT0"))) {
        VecAdd<<<grid_size, block_size, 0, stream_>>>(
            cuda_input0_, cuda_input1_, cuda_output_, batchn_elements);
      } else {
        VecSub<<<grid_size, block_size, 0, stream_>>>(
            cuda_input0_, cuda_input1_, cuda_output_, batchn_elements);
      }

      cudaError_t cuerr = cudaMemcpyAsync(
          obuffer, cuda_output_, batchn_byte_size, cudaMemcpyDeviceToHost,
          stream_);
      if (cuerr != cudaSuccess) {
        LOG_ERROR << "failed to copy output values from GPU for addsub: "
                  << cudaGetErrorString(cuerr);
        payload.error_code = kCudaMemcpy;
        break;
      }
    }
  }

  // Wait for all compute and memcpy to complete before returning.
  cudaStreamSynchronize(stream_);

  return kSuccess;
}

int
Context::Execute(
    const uint32_t payload_cnt, CustomPayload* payloads,
    CustomGetNextInputFn_t input_fn, CustomGetOutputFn_t output_fn)
{
  if (gpu_device_ == CUSTOM_NO_GPU_DEVICE) {
    return ExecuteCPU(payload_cnt, payloads, input_fn, output_fn);
  } else {
    return ExecuteGPU(payload_cnt, payloads, input_fn, output_fn);
  }
}

/////////////

extern "C" {

int
CustomInitialize(
    const char* serialized_model_config, int gpu_device_id,
    void** custom_context)
{
  // Convert the serialized model config to a ModelConfig object.
  ModelConfig model_config;
  if (!model_config.ParseFromString(serialized_model_config)) {
    return kInvalidModelConfig;
  }

  // Create the context and validate that the model configuration is
  // something that we can handle.
  Context* context = new Context(model_config, gpu_device_id);
  int err = context->Init();
  if (err != kSuccess) {
    return err;
  }

  *custom_context = static_cast<void*>(context);

  return kSuccess;
}

int
CustomFinalize(void* custom_context)
{
  if (custom_context != nullptr) {
    Context* context = static_cast<Context*>(custom_context);
    delete context;
  }

  return kSuccess;
}

const char*
CustomErrorString(void* custom_context, int errcode)
{
  switch (errcode) {
    case kSuccess:
      return "success";
    case kInvalidModelConfig:
      return "invalid model configuration";
    case kGpuNotSupported:
      return "execution on GPU not supported";
    case kInputOutputShape:
      return "model must have two inputs and two outputs with the same shape";
    case kInputName:
      return "model inputs must be named 'INPUT0' and 'INPUT1'";
    case kOutputName:
      return "model outputs must be named 'OUTPUT0' and 'OUTPUT1'";
    case kInputOutputDataType:
      return "model inputs and outputs must have TYPE_INT32 data-type";
    case kInputContents:
      return "unable to get input tensor values";
    case kInputSize:
      return "unexpected size for input tensor";
    case kOutputBuffer:
      return "unable to get buffer for output tensor values";
    case kCudaMalloc:
      return "cudaMalloc failed";
    case kCudaMemcpy:
      return "cudaMemcpy failed";
    default:
      break;
  }

  return "unknown error";
}

int
CustomExecute(
    void* custom_context, const uint32_t payload_cnt, CustomPayload* payloads,
    CustomGetNextInputFn_t input_fn, CustomGetOutputFn_t output_fn)
{
  if (custom_context == nullptr) {
    return kUnknown;
  }

  Context* context = static_cast<Context*>(custom_context);
  return context->Execute(payload_cnt, payloads, input_fn, output_fn);
}

}  // extern "C"

}}}}  // namespace nvidia::inferenceserver::custom::addsub
