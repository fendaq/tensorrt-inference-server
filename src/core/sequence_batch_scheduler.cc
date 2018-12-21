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

#include "src/core/sequence_batch_scheduler.h"

#include <sys/resource.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>
#include "src/core/constants.h"
#include "src/core/infer.h"
#include "src/core/logging.h"
#include "src/core/server_status.h"

namespace nvidia { namespace inferenceserver {

SequenceBatchScheduler::SequenceBatchScheduler(
    const ModelConfig& config, const uint32_t runner_cnt,
    StandardRunFunc OnSchedule)
{
  // Get the batch size to allow for each runner. This is 1 even if
  // the model doesn't support batching.
  size_t batch_size = std::max(1, config.max_batch_size());

  // Create one SequenceBatch object for each requested runner. The
  // SequenceBatch object has a thread that manages the batch of
  // requests.
  for (uint32_t c = 0; c < runner_cnt; ++c) {
    std::shared_ptr<SequenceBatch> sb =
        std::make_shared<SequenceBatch>(c, batch_size, config, OnSchedule);
    batches_.push_back(sb);
  }
}

void
SequenceBatchScheduler::Enqueue(
    const std::shared_ptr<ModelInferStats>& stats,
    const std::shared_ptr<InferRequestProvider>& request_provider,
    const std::shared_ptr<InferResponseProvider>& response_provider,
    std::function<void(tensorflow::Status)> OnComplete)
{
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);

  const auto& request_header = request_provider->RequestHeader();
  const CorrelationID correlation_id = request_header.correlation_id();

  // A request must have a correlation ID to be processed correctly by
  // this scheduler. A value of 0 (zero) indicates that the request
  // doesn't have a correlation ID.
  if (correlation_id == 0) {
    OnComplete(tensorflow::errors::InvalidArgument(
        "inference request to model '", request_provider->ModelName(),
        "' must specify a correlation ID"));
    return;
  }

  SequenceTarget* target = nullptr;

  std::unique_lock<std::mutex> lock(mu_);

  // If the request's correlation_id is new, then attempt to find a
  // free slot to use for that ID. If one doesn't exist then put the
  // request onto the backlog queue where it must wait for a slot to
  // come free. If a free slot is found assign this and subsequent
  // requests with this correlation ID to that same
  // SequenceBatch+slot.
  auto sb_itr = sequence_to_target_map_.find(correlation_id);
  if (sb_itr == sequence_to_target_map_.end()) {
    bool found_slot = false;
    std::shared_ptr<SequenceBatch> isb;
    uint32_t islot;

    // Look through the slots of the batches in order to find the
    // first free slot. This methods favors keeping the requests in
    // the minimum number of model instances and creating large
    // batches... another option would be to distribute the requests
    // across all the instances.
    for (const std::shared_ptr<SequenceBatch>& bsb : batches_) {
      found_slot = bsb->GetFreeSlot(&islot);
      if (found_slot) {
        isb = bsb;
        break;
      }
    }

    target = &sequence_to_target_map_[correlation_id];
    if (found_slot) {
      target->sequence_batch_ = isb;
      target->slot_ = islot;
    } else {
      backlog_sequence_ids_.push_back(correlation_id);
    }
  } else {
    target = &sb_itr->second;
  }

  if (target->IsBacklog()) {
    target->backlog_.emplace_back(
        now, stats, request_provider, response_provider, OnComplete);
    return;
  }

  std::shared_ptr<SequenceBatch> sb = target->sequence_batch_;
  uint32_t slot = target->slot_;

  lock.unlock();

  sb->Enqueue(
      slot, correlation_id, now, stats, request_provider, response_provider,
      OnComplete);
}

SequenceBatchScheduler::SequenceBatch::SequenceBatch(
    const uint32_t runner_id, const size_t batch_size,
    const ModelConfig& config, StandardRunFunc OnSchedule)
    : OnSchedule_(OnSchedule), scheduler_thread_exit_(false),
      scheduler_idle_(false), correlation_ids_(batch_size, 0),
      queues_(batch_size), max_active_slot_(-1)
{
  // Create a scheduler thread associated with 'runner_id' that
  // executes the queued payloads.
  const int nice = GetCpuNiceLevel(config);
  scheduler_thread_.reset(new std::thread(
      [this, runner_id, nice]() { SchedulerThread(runner_id, nice); }));
}

SequenceBatchScheduler::SequenceBatch::~SequenceBatch()
{
  // Signal the scheduler thread to exit...
  {
    std::unique_lock<std::mutex> lock(mu_);
    scheduler_thread_exit_ = true;
  }

  cv_.notify_one();
  scheduler_thread_->join();
}

bool
SequenceBatchScheduler::SequenceBatch::GetFreeSlot(uint32_t* slot)
{
  std::unique_lock<std::mutex> lock(mu_);

  // A slot is free if it doesn't have a correlation ID assigned to it
  // and there are no requests in the queue.
  for (size_t i = 0; i < queues_.size(); ++i) {
    if ((correlation_ids_[i] == 0) && queues_[i].empty()) {
      *slot = i;
      return true;
    }
  }

  return false;
}

void
SequenceBatchScheduler::SequenceBatch::Enqueue(
    const uint32_t slot, const CorrelationID correlation_id,
    const struct timespec queue_timestamp,
    const std::shared_ptr<ModelInferStats>& stats,
    const std::shared_ptr<InferRequestProvider>& request_provider,
    const std::shared_ptr<InferResponseProvider>& response_provider,
    std::function<void(tensorflow::Status)> OnComplete)
{
  bool wake_runner = false;

  {
    std::lock_guard<std::mutex> lock(mu_);

    correlation_ids_[slot] = correlation_id;
    queues_[slot].emplace_back(
        queue_timestamp, stats, request_provider, response_provider,
        OnComplete);
    max_active_slot_ = std::max(max_active_slot_, static_cast<int32_t>(slot));

    // If runner is idle then wake it to service this request. We do
    // the actual wake outside of the lock to avoid having the woken
    // thread immediately block on the lock
    wake_runner = scheduler_idle_;
  }

  if (wake_runner) {
    cv_.notify_one();
  }
}

void
SequenceBatchScheduler::SequenceBatch::SchedulerThread(
    const uint32_t runner_id, const int nice)
{
  if (setpriority(PRIO_PROCESS, syscall(SYS_gettid), nice) == 0) {
    LOG_VERBOSE(1) << "Starting sequence-batch scheduler thread " << runner_id
                   << " at nice " << nice << "...";
  } else {
    LOG_VERBOSE(1) << "Starting sequence-batch scheduler thread " << runner_id
                   << " at default nice (requested nice " << nice
                   << " failed)...";
  }

  // For debugging, delay start of thread until the queue contains the
  // specified number of entries.
  const char* dstr = getenv("TRTSERVER_DELAY_SCHEDULER");
  size_t delay_cnt = 0;
  if (dstr != nullptr) {
    delay_cnt = atoi(dstr);
    LOG_INFO << "Delaying scheduler thread " << runner_id << " until "
             << delay_cnt << " queued payloads...";
  }

  const uint64_t default_wait_microseconds = 500 * 1000;

  while (!scheduler_thread_exit_) {
    std::shared_ptr<std::vector<Scheduler::Payload>> payloads;
    uint64_t wait_microseconds = 0;

    // Hold the lock for as short a time as possible.
    {
      std::unique_lock<std::mutex> lock(mu_);
      if (delay_cnt > 0) {
        wait_microseconds = 10 * 1000;
        // Debugging... wait until queues together contain at least
        // 'delay_cnt' items...
        size_t total_size = 0;
        for (const auto& q : queues_) {
          total_size += q.size();
        }
        if (total_size >= delay_cnt) {
          delay_cnt = 0;
        }
      } else {
        // Make sure there is at least one request that needs to be
        // handled.
        bool have_request = false;
        for (int32_t slot = 0; slot <= max_active_slot_; ++slot) {
          if ((correlation_ids_[slot] != 0) && !queues_[slot].empty()) {
            have_request = true;
            break;
          }
        }

        if (!have_request) {
          wait_microseconds = default_wait_microseconds;
        } else {
          // Collect payloads from slot 0 to max_active_slot_.
          for (int32_t slot = 0; slot <= max_active_slot_; ++slot) {
            // If 'slot' is not active then need to send a dummy payload
            // that "pads" batch so that other payloads stay in the
            // correct slot.
            if (correlation_ids_[slot] == 0) {
              continue;
            }

            std::deque<SequencePayload>& queue = queues_[slot];

            // If 'slot' is active but doesn't have any requests then
            // need to send a dummy payload that "pads"...
            if (queue.empty()) {
              continue;
            }

            const SequencePayload& slot_payload = queue.front();

            payloads->emplace_back(
                slot_payload.queued_timestamp_, slot_payload.stats_,
                slot_payload.request_provider_, slot_payload.response_provider_,
                slot_payload.complete_function_);

            queue.pop_front();
          }
        }
      }

      // If no requests are to be handled, wait for notification or
      // for the specified timeout before checking the queues again.
      if (wait_microseconds > 0) {
        scheduler_idle_ = true;
        std::chrono::microseconds wait_timeout(wait_microseconds);
        cv_.wait_for(lock, wait_timeout);
        scheduler_idle_ = false;
      }
    }

    if ((payloads != nullptr) && !payloads->empty()) {
      auto OnCompleteQueuedPayloads = [payloads](tensorflow::Status status) {
        bool found_success = false;
        for (auto& payload : *payloads) {
          tensorflow::Status final_status =
              status.ok() ? (payload.status_.ok() ? payload.compute_status_
                                                  : payload.status_)
                          : status;

          // All the payloads executed together, so count 1 execution in
          // the first successful payload. Other payloads stay at 0
          // executions.
          if (!found_success && final_status.ok()) {
            payload.stats_->SetModelExecutionCount(1);
            found_success = true;
          }
          payload.complete_function_(final_status);
        }
      };

      OnSchedule_(runner_id, payloads.get(), OnCompleteQueuedPayloads);
    }
  }  // end runner loop

  LOG_VERBOSE(1) << "Stopping sequence-batch scheduler thread " << runner_id
                 << "...";
}

}}  // namespace nvidia::inferenceserver
