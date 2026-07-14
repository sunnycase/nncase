// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.Schedule.TileGraph;

/// <summary>
/// Classifies logical buffer-alias endpoints independently from physical
/// storage placement. Pure buffer views may create descriptors, but their
/// source and result endpoints never own explicit-copy storage.
/// </summary>
internal static class TileBufferAliasAnalysis
{
    public static bool IsPureAliasSource(BufferIdentity buffer)
        => !buffer.IsOutput &&
           buffer.Node.IsPureBufferView &&
           buffer.Node.BufferAliases.Any(alias => alias.SourceAccessIndex == buffer.Index);

    public static bool IsPureAliasResult(BufferIdentity buffer)
        => buffer.IsOutput &&
           buffer.Node.IsPureBufferView &&
           buffer.Node.BufferAliases.Any(alias => alias.ResultAccessIndex == buffer.Index);

    public static bool IsPureAliasEndpoint(BufferIdentity buffer)
        => IsPureAliasSource(buffer) || IsPureAliasResult(buffer);
}
