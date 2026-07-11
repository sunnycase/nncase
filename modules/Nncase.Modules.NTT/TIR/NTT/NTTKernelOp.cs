// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using Nncase.IR;

namespace Nncase.TIR.NTT;

public abstract class NTTKernelOp : Op
{
    public override IReadOnlyList<ParameterInfo> Parameters
    {
        get
        {
            var parameters = base.Parameters;
            var missing = parameters.Where(parameter => parameter.MemoryEffect is null).ToArray();
            if (missing.Length != 0)
            {
                throw new InvalidOperationException(
                    $"{GetType().Name} must declare a memory effect for every operand. Missing: {string.Join(", ", missing.Select(parameter => parameter.Name))}.");
            }

            return parameters;
        }
    }
}
