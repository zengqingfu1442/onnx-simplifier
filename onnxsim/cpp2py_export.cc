/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <nanobind/nanobind.h>
#include <nanobind/trampoline.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <string>
#include <tuple>
#include <vector>

#include "onnx/defs/schema.h"
#include "onnx/proto_utils.h"
#include "onnxoptimizer/optimize.h"
#include "onnxsim.h"

namespace py = nanobind;
using namespace nanobind::literals;

namespace {

using onnx::OpSchema;

// A formal parameter (input/output) as marshalled from the Python ``onnx``
// module: (name, description, type_str, option, is_homogeneous, min_arity).
// ``option`` is the integer value of onnx's FormalParameterOption enum
// (Single=0, Optional=1, Variadic=2).
using PyFormalParameter =
    std::tuple<std::string, std::string, std::string, int, bool, int>;
// An attribute: (name, description, type, required, default_value). ``type`` is
// the integer value of onnx's AttributeProto::AttributeType enum. When
// ``default_value`` is a non-empty serialized AttributeProto the attribute is
// optional with that default; when empty, ``required`` decides.
using PyAttribute = std::tuple<std::string, std::string, int, bool, py::bytes>;
// A type constraint: (type_param_str, allowed_type_strs, description).
using PyTypeConstraint =
    std::tuple<std::string, std::vector<std::string>, std::string>;

// Ensure ``domain`` exists in the schema registry's domain-to-version range and
// that ``version`` falls inside it, so a schema with that since_version can be
// registered. The default ONNX domain ("") is always present; custom domains
// coming from user-registered schemas usually are not, and onnx refuses to
// register a schema whose domain/version is outside the known range.
void EnsureDomainVersion(const std::string& domain, int version) {
  auto& range = onnx::OpSchemaRegistry::DomainToVersionRange::Instance();
  const auto& map = range.Map();
  auto it = map.find(domain);
  if (it == map.end()) {
    range.AddDomainToVersion(domain, /*min_version=*/std::min(version, 1),
                             /*max_version=*/std::max(version, 1));
  } else {
    const int lo = std::min(it->second.first, version);
    const int hi = std::max(it->second.second, version);
    if (lo != it->second.first || hi != it->second.second) {
      range.UpdateDomainToVersion(domain, lo, hi);
    }
  }
}

// The default ONNX domain is stored as the empty string in the schema registry;
// "ai.onnx" is an accepted spelling of the same domain.
std::string NormalizeDomain(const std::string& domain) {
  return domain == "ai.onnx" ? std::string() : domain;
}

}  // namespace

struct PyModelExecutor : public ModelExecutor {
  using ModelExecutor::ModelExecutor;

  std::vector<onnx::TensorProto> _Run(
      const onnx::ModelProto& model,
      const std::vector<onnx::TensorProto>& inputs) const override {
    std::vector<py::bytes> inputs_bytes;
    std::transform(inputs.begin(), inputs.end(),
                   std::back_inserter(inputs_bytes),
                   [](const onnx::TensorProto& x) {
                     const std::string str = x.SerializeAsString();
                     return py::bytes(str.data(), str.size());
                   });
    std::string model_str = model.SerializeAsString();
    auto output_bytes = _PyRun(py::bytes(model_str.data(), model_str.size()), inputs_bytes);
    std::vector<onnx::TensorProto> output_tps;
    std::transform(output_bytes.begin(), output_bytes.end(),
                   std::back_inserter(output_tps), [](const py::bytes& x) {
                     onnx::TensorProto tp;
                     tp.ParseFromString(std::string(x.c_str(), x.size()));
                     return tp;
                   });
    return output_tps;
  }

  virtual std::vector<py::bytes> _PyRun(
      const py::bytes& model_bytes,
      const std::vector<py::bytes>& inputs_bytes) const = 0;
};

struct PyModelExecutorTrampoline : public PyModelExecutor {
  NB_TRAMPOLINE(PyModelExecutor, 1);

  /* Inherit the constructors */
  // using PyModelExecutor::PyModelExecutor;

  /* Trampoline (need one for each virtual function) */
  std::vector<py::bytes> _PyRun(
      const py::bytes& model_bytes,
      const std::vector<py::bytes>& inputs_bytes) const override {
    NB_OVERRIDE_PURE_NAME(
        "Run", _PyRun, /* Name of function in C++ (must match Python name) */
        model_bytes, inputs_bytes /* Argument(s) */
    );
  }
};

NB_MODULE(onnxsim_cpp2py_export, m) {
  m.doc() = "ONNX Simplifier";

  using namespace py::literals;

  m.def("simplify",
        [](std::shared_ptr<PyModelExecutor> executor,
           const py::bytes& model_proto_bytes,
           std::optional<std::vector<std::string>> skip_optimizers,
           bool constant_folding, bool shape_inference,
           size_t tensor_size_threshold) -> py::bytes {
          // force env initialization to register opset
          InitEnv();
          ONNX_NAMESPACE::ModelProto model;
          ParseProtoFromBytes(&model, model_proto_bytes.c_str(), model_proto_bytes.size());
          auto const result = Simplify(*executor, model, skip_optimizers,
                                       constant_folding, shape_inference,
                                       tensor_size_threshold);
          std::string out;
          result.SerializeToString(&out);
          return py::bytes(out.data(), out.size());
        }, "executor"_a, "model_bytes"_a, "skip_optimizers"_a.none(),
        "constant_folding"_a = true, "shape_inference"_a = true, "tensor_size_threshold"_a)
      .def("simplify_path",
           [](std::shared_ptr<PyModelExecutor> executor,
              const std::string& in_path, const std::string& out_path,
              std::optional<std::vector<std::string>> skip_optimizers,
              bool constant_folding, bool shape_inference,
              size_t tensor_size_threshold) -> bool {
             // force env initialization to register opset
             InitEnv();
             SimplifyPath(*executor, in_path, out_path, skip_optimizers,
                          constant_folding, shape_inference,
                          tensor_size_threshold);
             return true;
           }, "executor"_a, "in_path"_a, "out_path"_a,
           "skip_optimizers"_a.none(),
           "constant_folding"_a = true, "shape_inference"_a = true,
           "tensor_size_threshold"_a)
      .def("_list_optimizers",
           []() {
            py::list ret;
            for (const auto& p : onnx::optimization::GetFuseAndEliminationPass()) {
              ret.append(p);
            }
            return ret;
           })
      // Whether onnxsim's internal (statically linked) schema registry already
      // knows an operator, at any opset version, in ``domain``. Used to skip
      // operators that do not need importing from the Python ``onnx`` module.
      .def("_has_schema",
           [](const std::string& op_type, const std::string& domain) -> bool {
             return onnx::OpSchemaRegistry::Schema(
                        op_type, NormalizeDomain(domain)) != nullptr;
           }, "op_type"_a, "domain"_a)
      // Register a single operator schema into onnxsim's internal schema
      // registry. onnxsim links its own copy of ONNX, so its registry is
      // separate from the one the Python ``onnx`` module uses; this bridges a
      // schema (e.g. one a user added via ``onnx.defs.register_schema``) across
      // that boundary so the model passes ``check_model`` (GitHub issue #326).
      //
      // The schema is registered without a type/shape inference function -- such
      // a function is a native pointer in the Python ``onnx`` library and cannot
      // be called from onnxsim's separately linked copy -- so shape inference
      // simply flows past the operator instead of stopping at it. Registration
      // never raises: a malformed or duplicate schema is reported to stderr and
      // ignored, matching the other schema-registration paths in onnxsim.
      .def("_register_schema",
           [](const std::string& name, const std::string& domain,
              int since_version, const std::string& doc,
              const std::vector<PyFormalParameter>& inputs,
              const std::vector<PyFormalParameter>& outputs,
              const std::vector<PyAttribute>& attributes,
              const std::vector<PyTypeConstraint>& type_constraints) {
             if (since_version < 1) {
               since_version = 1;
             }
             const std::string dom = NormalizeDomain(domain);
             EnsureDomainVersion(dom, since_version);

             OpSchema schema;
             schema.SetName(name).SetDomain(dom).SinceVersion(since_version).SetDoc(
                 doc);

             int idx = 0;
             for (const auto& p : inputs) {
               schema.Input(
                   idx++, std::get<0>(p), std::get<1>(p), std::get<2>(p),
                   static_cast<OpSchema::FormalParameterOption>(std::get<3>(p)),
                   std::get<4>(p), std::get<5>(p));
             }
             idx = 0;
             for (const auto& p : outputs) {
               schema.Output(
                   idx++, std::get<0>(p), std::get<1>(p), std::get<2>(p),
                   static_cast<OpSchema::FormalParameterOption>(std::get<3>(p)),
                   std::get<4>(p), std::get<5>(p));
             }
             for (const auto& a : attributes) {
               const py::bytes& default_value = std::get<4>(a);
               if (default_value.size() > 0) {
                 onnx::AttributeProto proto;
                 proto.ParseFromArray(default_value.c_str(),
                                      static_cast<int>(default_value.size()));
                 schema.Attr(OpSchema::Attribute(std::get<0>(a), std::get<1>(a),
                                                 std::move(proto)));
               } else {
                 schema.Attr(OpSchema::Attribute(
                     std::get<0>(a), std::get<1>(a),
                     static_cast<onnx::AttributeProto::AttributeType>(
                         std::get<2>(a)),
                     std::get<3>(a)));
               }
             }
             for (const auto& tc : type_constraints) {
               schema.TypeConstraint(std::get<0>(tc), std::get<1>(tc),
                                     std::get<2>(tc));
             }

             onnx::RegisterSchema(std::move(schema), /*opset_version_to_load=*/0,
                                  /*fail_duplicate_schema=*/false,
                                  /*fail_with_exception=*/false);
           },
           "name"_a, "domain"_a, "since_version"_a, "doc"_a, "inputs"_a,
           "outputs"_a, "attributes"_a, "type_constraints"_a)
      ;

  py::class_<PyModelExecutor, PyModelExecutorTrampoline>(m, "ModelExecutor")
      .def(py::init<>())
      .def("Run", &PyModelExecutor::_PyRun);
}
