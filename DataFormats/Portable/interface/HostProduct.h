#ifndef DataFormats_Common_interface_HostProduct_H
#define DataFormats_Common_interface_HostProduct_H

#include <alpaka/alpaka.hpp>
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

// a heterogeneous unique pointer...
template <typename T>
class HostProduct {
public:
  HostProduct() = default;  // make root happy
  ~HostProduct() = default;
  HostProduct(HostProduct&&) = default;
  HostProduct& operator=(HostProduct&&) = default;

  explicit HostProduct(cms::alpakatools::host_buffer<T>&& p) : hm_ptr(std::move(p)) {}
  explicit HostProduct(std::unique_ptr<T>&& p) : std_ptr(std::move(p)) {}

  auto const* get() const { return std_ptr ? std_ptr.get() : hm_ptr.data(); }

  auto const& operator*() const { return *get(); }

  auto const* operator->() const { return get(); }

private:
  cms::alpakatools::host_buffer<T> hm_ptr;  //!
  std::unique_ptr<T> std_ptr;               //!
};

#endif