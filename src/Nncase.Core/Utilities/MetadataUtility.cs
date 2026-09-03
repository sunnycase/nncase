// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Immutable;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Fx = System.Func<Nncase.IR.Expr, Nncase.IR.Expr>;
using ParameterInfo = Nncase.IR.ParameterInfo;
using Tuple = Nncase.IR.Tuple;

namespace Nncase.Utilities;

/// <summary>
/// Metadata Utility.
/// </summary>
///
public static class MetadataUtility
{
    /// <summary>
    /// Inherit MetaData.
    /// </summary>
    ///
    public static T InheritMetaData<T>(this T newCall, BaseExpr oldCall)
        where T : BaseExpr
    {
        if (oldCall.Metadata.OutputNames is not null)
        {
            newCall.Metadata.OutputNames = oldCall.Metadata.OutputNames;
        }

        // These fields identify the expression's model-level semantics and
        // placement, so semantics-preserving rewrites must carry them forward.
        // Range and microkernel metadata are derived for a concrete expression
        // and must be recomputed after rewriting.
        if (oldCall.Metadata.SemanticRegion is not null)
        {
            newCall.Metadata.SemanticRegion = oldCall.Metadata.SemanticRegion;
        }

        if (oldCall.Metadata.ExecutionModuleKind is not null)
        {
            newCall.Metadata.ExecutionModuleKind = oldCall.Metadata.ExecutionModuleKind;
        }

        return newCall;
    }
}
