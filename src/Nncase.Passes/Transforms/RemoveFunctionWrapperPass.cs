// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.TIR;

namespace Nncase.Passes.Transforms;

/// <summary>
/// Resolve lowered function wrappers in TIR bodies to direct prim function calls.
/// </summary>
public sealed class RemoveFunctionWrapperPass : ModulePass
{
    /// <inheritdoc/>
    protected override Task<IRModule> RunCoreAsync(IRModule input, RunPassContext context)
    {
        var resolver = new FunctionResolver(input);
        foreach (var primFunction in input.Functions.OfType<PrimFunction>().ToArray())
        {
            var rewriter = new FunctionWrapperCallRewriter(resolver);
            rewriter.Rewrite(primFunction);

            if (rewriter.IsMutated && !CompilerServices.InferenceType(primFunction))
            {
                throw new InvalidOperationException($"Type inference failed after removing function wrappers in {primFunction.Name}.");
            }
        }

        return Task.FromResult(input);
    }

    private sealed class FunctionResolver
    {
        private readonly Dictionary<BaseFunction, PrimFunction> _functions = new(ReferenceEqualityComparer.Instance);
        private readonly Dictionary<(string Name, string ModuleKind), PrimFunction> _loweredFunctions = new();

        public FunctionResolver(IRModule module)
        {
            foreach (var function in module.Functions)
            {
                switch (function)
                {
                    case PrimFunction primFunction:
                        Register(function, primFunction);
                        break;
                    case PrimFunctionWrapper primFunctionWrapper:
                        Register(function, primFunctionWrapper.Target);
                        RegisterLoweredFunction(function.Name, primFunctionWrapper.Target.ModuleKind, primFunctionWrapper.Target);
                        break;
                }
            }

            foreach (var function in module.Functions.OfType<Function>())
            {
                if (_functions.ContainsKey(function))
                {
                    continue;
                }

                if (TryResolveHighLevelFunction(function.Name, function.ModuleKind, out var primFunction))
                {
                    Register(function, primFunction);
                }
            }
        }

        public bool TryResolve(BaseFunction function, out PrimFunction primFunction)
        {
            while (function is FunctionWrapper wrapper)
            {
                function = wrapper.Target;
            }

            if (_functions.TryGetValue(function, out primFunction!))
            {
                return true;
            }

            if (function is Function highLevelFunction
                && TryResolveHighLevelFunction(highLevelFunction.Name, highLevelFunction.ModuleKind, out primFunction!))
            {
                Register(highLevelFunction, primFunction);
                return true;
            }

            primFunction = null!;
            return false;
        }

        private bool TryResolveHighLevelFunction(string functionName, string moduleKind, out PrimFunction primFunction)
        {
            if (_loweredFunctions.TryGetValue((functionName, moduleKind), out primFunction!))
            {
                return true;
            }

            primFunction = null!;
            return false;
        }

        private void Register(BaseFunction function, PrimFunction primFunction)
        {
            _functions[function] = primFunction;
        }

        private void RegisterLoweredFunction(string name, string moduleKind, PrimFunction primFunction)
        {
            var key = (name, moduleKind);
            if (_loweredFunctions.TryGetValue(key, out var existing) && !ReferenceEquals(existing, primFunction))
            {
                throw new InvalidOperationException($"Function wrapper resolver found ambiguous prim functions for {name} in module {moduleKind}.");
            }

            _loweredFunctions[key] = primFunction;
        }
    }

    private sealed class FunctionWrapperCallRewriter : ExprRewriter
    {
        private readonly FunctionResolver _resolver;

        public FunctionWrapperCallRewriter(FunctionResolver resolver)
            : base(visitOtherFunctions: false)
        {
            _resolver = resolver;
        }

        protected override BaseExpr RewriteLeafCall(Call expr)
        {
            if (expr.Target is FunctionWrapper functionWrapper)
            {
                if (!_resolver.TryResolve(functionWrapper, out var primFunction))
                {
                    throw new InvalidOperationException($"Cannot resolve function wrapper {functionWrapper.Name} targeting {functionWrapper.Target.Name} to a prim function before bufferize.");
                }

                return expr.With(target: primFunction, arguments: FlattenTupleArguments(expr.Arguments.ToArray()));
            }

            if (expr.Target is PrimFunctionWrapper primFunctionWrapper)
            {
                return expr.With(target: primFunctionWrapper.Target, arguments: FlattenTupleArguments(expr.Arguments.ToArray()));
            }

            if (expr.Target is Function function && _resolver.TryResolve(function, out var directPrimFunction))
            {
                return expr.With(target: directPrimFunction, arguments: FlattenTupleArguments(expr.Arguments.ToArray()));
            }

            return expr;
        }

        private static BaseExpr[] FlattenTupleArguments(BaseExpr[] arguments)
            => arguments.SelectMany(argument => argument is IR.Tuple tuple ? tuple.Fields.ToArray() : [argument]).ToArray();
    }
}
