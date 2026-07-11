// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.IR.Buffers;

/// <summary>
/// Allocate buffer view.
/// </summary>
public sealed partial class AllocateBufferView : Op
{
    /// <summary>
    /// Get the input parameter.
    /// </summary>
    public static readonly ParameterInfo Buffer = new(typeof(AllocateBufferView), 0, "buffer");

    /// <summary>
    /// Gets the logical offset within the buffer's local distributed shard.
    /// The view's global origin is the shard origin plus this offset.
    /// </summary>
    public static readonly ParameterInfo LocalOffset = new(typeof(AllocateBufferView), 1, "local_offset", IsShapeType());

    /// <inheritdoc/>
    public override bool CanFoldConstCall => false;
}
