// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.IR.Tensors
{
    public sealed record SpaceToBatch() : Op
    {
        public static readonly ParameterInfo Input = new(typeof(SpaceToBatch), 0, "input");

        public static readonly ParameterInfo BlockShape = new(typeof(SpaceToBatch), 1, "block_shape");

        public static readonly ParameterInfo Paddings = new(typeof(SpaceToBatch), 2, "paddings");

        public IRType InferInvokeResultType(ITypeInferenceContext context)
        {
            throw new NotImplementedException();
        }
    }
}
