// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.Passes;
using Nncase.Passes.Mutators;
using Nncase.Tests.TestFixture;
using Nncase.TIR;
using Xunit;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.Tensors;

namespace Nncase.Tests.TIRTest;

[AutoSetupTestMethod(InitSession = true)]
public sealed class UnitTestMutators : TestClassBase
{
    public UnitTestMutators()
    {
#if DEBUG
        CompileOptions.DumpFlags = Diagnostics.DumpFlags.Rewrite | Diagnostics.DumpFlags.PassIR;
#endif
    }

    [Fact]
    public void TestFlattenSequentialPreservesTraceScopeWithoutReportingMutation()
    {
        var traced = new Sequential(new Expr[] { T.Nop() }, "fusion[op0:test]");
        var outer = new Sequential(traced);
        var rewriter = new FlattenSequential();

        var rewritten = rewriter.Rewrite(outer);

        Assert.Same(outer, rewritten);
        Assert.False(rewriter.IsMutated);
        Assert.Same(traced, Assert.Single(outer.Fields.ToArray()));
    }

    [Fact]
    public void TestUnrollCleanupPreservesTraceScopeWithoutReportingMutation()
    {
        var traced = new Sequential(new Expr[] { T.Nop() }, "fusion[op0:test]");
        var outer = new Sequential(traced);
        var rewriter = new UnRollLoopSequential();

        var rewritten = rewriter.Rewrite(outer);

        Assert.Same(outer, rewritten);
        Assert.False(rewriter.IsMutated);
        Assert.Same(traced, Assert.Single(outer.Fields.ToArray()));
    }

    [Fact]
    public void TestFlattenSequentialPreservesCodegenScopeWithoutReportingMutation()
    {
        var scoped = T.CodegenScope("fusion[op0:test]", new Sequential(T.Nop()));
        var outer = new Sequential(scoped);
        var rewriter = new FlattenSequential();

        var rewritten = rewriter.Rewrite(outer);

        Assert.Same(outer, rewritten);
        Assert.False(rewriter.IsMutated);
        Assert.Same(scoped, Assert.Single(outer.Fields.ToArray()));
        Assert.True(scoped.PreserveCodegenBoundary);
    }

    [Fact]
    public void TestTailLoopPeelingPreservesFreeBufferVarIdentity()
    {
        var inputType = new TensorType(DataTypes.Float32, new[] { 5 });
        var input = new BufferVar("input", inputType, BufferVarRole.Input, MemoryLocation.Input);
        var body = T.Sequential(
            T.Serial(out var i, (0, 5, 4)).Body(
                T.Let(
                    out var view,
                    IR.F.Buffer.BufferSubview(input, new RankedShape(new Dimension[] { i }), new RankedShape(new Dimension[] { 1 }))).Body(
                    T.Nop())));
        var function = new PrimFunction("tail_loop", Callable.CPUModuleKind, body, new[] { input });

        var rewritten = (PrimFunction)new TailLoopPeeling().Rewrite(function);
        var bodyBufferVars = ExprCollector.Collect(rewritten.Body).OfType<BufferVar>().ToArray();
        var loops = ExprCollector.Collect(rewritten.Body).OfType<TIR.For>().ToArray();
        var fullView = Assert.Single(ExprCollector.Collect(loops[0].Body).OfType<Let>()).Var;
        var tailView = Assert.Single(ExprCollector.Collect(loops[1].Body).OfType<Let>()).Var;

        Assert.NotEmpty(bodyBufferVars);
        Assert.All(bodyBufferVars, bufferVar => Assert.Same(rewritten.Parameters[0], bufferVar));
        Assert.NotSame(fullView, tailView);
        Assert.Equal(2, loops.Length);
        Assert.Equal(LoopPartition.Full, loops[0].Partition);
        Assert.Equal(LoopPartition.Tail, loops[1].Partition);
        Assert.Equal(4, loops[0].Domain.Stop.FixedValue);
        Assert.Equal(4, loops[1].Domain.Start.FixedValue);

        var secondRewriter = new TailLoopPeeling();
        Assert.Same(rewritten, secondRewriter.Rewrite(rewritten));
        Assert.False(secondRewriter.IsMutated);
    }

    [Fact]
    public void TestTailLoopPeelingPreservesPhysicalStorageIdentity()
    {
        var inputType = new TensorType(DataTypes.Float32, new[] { 5 });
        var input = new BufferVar("input", inputType, BufferVarRole.Input, MemoryLocation.Input);
        var physicalBuffer = new PhysicalBuffer(64, input, 20, MemoryLocation.Input);
        var body = T.Sequential(
            T.Serial(out var i, (0, 5, 4)).Body(
                T.Let(
                    out _,
                    IR.F.Buffer.AllocateBufferView(
                        new TIR.Buffer(
                            "input_tile",
                            DataTypes.Float32,
                            new MemSpan(physicalBuffer, i * 4, 4),
                            new Dimension[] { 1 },
                            new Dimension[] { 1 },
                            null),
                        new RankedShape(0))).Body(
                    T.Nop())));
        var function = new PrimFunction("tail_loop", Callable.CPUModuleKind, body, new[] { input });

        var rewritten = (PrimFunction)new TailLoopPeeling().Rewrite(function);
        var physicalBuffers = ExprCollector.Collect(rewritten.Body).OfType<PhysicalBuffer>().ToArray();
        var loops = ExprCollector.Collect(rewritten.Body).OfType<TIR.For>().ToArray();
        var fullBuffer = Assert.Single(ExprCollector.Collect(loops[0].Body).OfType<TIR.Buffer>());
        var tailBuffer = Assert.Single(ExprCollector.Collect(loops[1].Body).OfType<TIR.Buffer>());

        Assert.Same(physicalBuffer, Assert.Single(physicalBuffers));
        Assert.Contains(loops[0].LoopVar, ExprCollector.Collect(fullBuffer.MemSpan.Start));
        Assert.Contains(loops[1].LoopVar, ExprCollector.Collect(tailBuffer.MemSpan.Start));
        Assert.DoesNotContain(loops[0].LoopVar, ExprCollector.Collect(tailBuffer.MemSpan.Start));
    }

    [Fact]
    public void TestNestedTailLoopPeelingProducesLinearBoundarySlabs()
    {
        var i = new DimVar("i");
        var j = new DimVar("j");
        var inner = new TIR.For(j, new TIR.Range(0, 7, 4), LoopMode.Serial, new Sequential(T.Nop()));
        var outer = new TIR.For(i, new TIR.Range(0, 5, 4), LoopMode.Serial, new Sequential(inner));

        var rewritten = Assert.IsType<Sequential>(new TailLoopPeeling().Rewrite(outer));
        var outerLoops = rewritten.Fields.ToArray().Select(Assert.IsType<TIR.For>).ToArray();
        var fullOuterInnerLoops = ExprCollector.Collect(outerLoops[0].Body).OfType<TIR.For>().ToArray();
        var tailOuterInnerLoops = ExprCollector.Collect(outerLoops[1].Body).OfType<TIR.For>().ToArray();

        Assert.Equal(2, outerLoops.Length);
        Assert.Equal(2, fullOuterInnerLoops.Length);
        Assert.Single(tailOuterInnerLoops);
        Assert.NotSame(fullOuterInnerLoops[0].LoopVar, tailOuterInnerLoops[0].LoopVar);
        Assert.Equal(7, tailOuterInnerLoops[0].Domain.Stop.FixedValue);
    }

    [Fact]
    public void TestDynamicReductionTailLoopPeelingPreservesModeAndBounds()
    {
        var stop = new DimVar("stop");
        var reductionAxis = new DimVar("k");
        var loop = new TIR.For(
            reductionAxis,
            new TIR.Range(0, stop, 8),
            LoopMode.Reduction,
            new Sequential(T.Nop()));

        var rewritten = Assert.IsType<Sequential>(new TailLoopPeeling().Rewrite(loop));
        var loops = rewritten.Fields.ToArray().Select(Assert.IsType<TIR.For>).ToArray();

        Assert.Equal(2, loops.Length);
        Assert.All(loops, peeled => Assert.Equal(LoopMode.Reduction, peeled.Mode));
        Assert.Equal(LoopPartition.Full, loops[0].Partition);
        Assert.Equal(LoopPartition.Tail, loops[1].Partition);
        Assert.Equal(loops[0].Domain.Stop, loops[1].Domain.Start);
        Assert.Same(stop, loops[1].Domain.Stop);
        Assert.Equal(8, loops[0].Domain.Step.FixedValue);
        Assert.Equal(8, loops[1].Domain.Step.FixedValue);
    }

    [Fact]
    public void TestEliminateStaticallyEmptyTailLoop()
    {
        var loop = new TIR.For(
            new DimVar("i_tail"),
            new TIR.Range(2, 2, 2),
            LoopMode.Serial,
            new Sequential(T.Nop()),
            LoopPartition.Tail);
        var rewriter = new EliminateEmptyLoops();

        var rewritten = rewriter.Rewrite(loop);

        Assert.True(rewriter.IsMutated);
        Assert.IsType<Call>(rewritten);
        Assert.IsType<Nop>(Assert.IsType<Call>(rewritten).Target);
    }

    [Fact]
    public async Task TestFoldConstCallWithTuple()
    {
        T.CreateBufferVar(new TensorType(DataTypes.BFloat16, new[] { 48 }), out var ddr_if);
        T.CreateBuffer(new TensorType(DataTypes.BFloat16, new[] { 9 }), MemoryLocation.Data, out var glb_if_ping);
        T.CreateBuffer(new TensorType(DataTypes.BFloat16, new[] { 9 }), MemoryLocation.Data, out var glb_if_pong);
        PrimFunction main;
        {
            main = T.PrimFunc("main", Callable.CPUModuleKind, ddr_if).Body(
                   T.Unrolled(out var w, (0, 48, 9)).Body(
                     new Call(
                         new LoadT(),
                         new BufferRegion(
                           ddr_if,
                           new Range[] { (w, Dimension.Min(w + 9L, 48L)) }),
                         GetItem(
                             new Tuple(new[] {
                          new BufferRegion(
                              glb_if_ping,
                              new Range[] { (0, Dimension.Min(w + 9L, 48L) - w) }),
                          new BufferRegion(
                              glb_if_pong,
                              new Range[] { (0, Dimension.Min(w + 9L, 48L) - w) }),
                          }),
                             w / 9L % 2L))))
           .Build();
        }

        var prmg = CompileSession.CreatePassManager("prmg");
        prmg.AddWithName<PrimFuncPass>("unroll").Configure(
          p =>
          {
              p.Add<UnRollLoopSequential>();
              p.Add<FoldConstCall>();
          });

        var module = new IR.IRModule(main);
        await prmg.RunAsync(module);

        var post = (TIR.PrimFunction)module.Entry!;
        {
            var getBuffer = (int i, ParameterInfo info) =>
            {
                var bufferRegion = (BufferRegion)((Call)post.Body.Fields[i])[info];
                return (TIR.Buffer)bufferRegion.Buffer;
            };
            int count = 0;
            for (int w = 0; w < 48; w += 9)
            {
                // Assert.True(object.ReferenceEquals(getBuffer(count, LoadT.DdrPp), post.Parameters[0]));
                var name = getBuffer(count++, LoadT.GlbPp).Name[^4..];

                // System.Console.WriteLine($"{w} {name}");
                if ((w / 9 % 2) == 0)
                {
                    Assert.Equal("ping", name);
                }
                else
                {
                    Assert.Equal("pong", name);
                }
            }
        }
    }

    [Fact]
    public async Task TestUnRollLoopSequential()
    {
        var main = T.PrimFunc("main", Callable.CPUModuleKind).Body(// (*i8) -> ()
            T.Unrolled(out var i, (0, 32, 4)).Body(// ()
              T.Unrolled(out var j, (0, 16, 4)).Body(// ()
                T.Unrolled(out var k, (0, 18, 6)).Body(// ()
                  T.Unrolled(out var l, (0, 32, 16)).Body(// ()
                    T.Block("block").Body()))))).Build();

        CompilerServices.InferenceType(main);
        var prmg = CompileSession.CreatePassManager("prmg");
        prmg.AddWithName<PrimFuncPass>("unroll").Configure(
          p =>
          {
              p.Add<UnRollLoopSequential>();
          });

        var module = new IR.IRModule(main);
        await prmg.RunAsync(module);
        Assert.Equal(32 / 4 * (16 / 4) * (18 / 6) * (32 / 16), ((TIR.PrimFunction)module.Entry!).Body.Count);
    }

    [Fact]
    public async Task TestUnRollLoopSequential2()
    {
        T.CreateBuffer(new TensorType(DataTypes.BFloat16, new[] { 3, 16, 24, 24 }), MemoryLocation.Input, out var ddr_if);
        T.CreateBuffer(new TensorType(DataTypes.BFloat16, new[] { 3, 10, 5, 9 }), MemoryLocation.Data, out var glb_if);

        PrimFunction main;
        {
            main = T.PrimFunc("main", Callable.CPUModuleKind).Body(
             T.Unrolled(out var n, (0, 3, 3)).Body(
               T.Unrolled(out var c, (0, 16, 10)).Body(
                 T.Unrolled(out var h, (0, 24, 5)).Body(
                   T.Unrolled(out var w, (0, 24, 9)).Body(
                     new Call(
                         new LoadT(),
                         new BufferRegion(
                           ddr_if,
                           new TIR.Range[] { (n, Dimension.Min(n + 3L, 3L)),
                                        (c, Dimension.Min(c + 10L, 16L)),
                                        (h, Dimension.Min(h + 5L, 24L)),
                                        (w, Dimension.Min(w + 9L, 24L)),
                                         }),
                         new BufferRegion(
                           glb_if,
                           new TIR.Range[] { (0, IR.Dimension.Min(n + 3L, 3L) - n),
                                        (0, Dimension.Min(c + 10L, 16L) - c),
                                        (0, Dimension.Min(h + 5L, 24L) - h),
                                        (0, Dimension.Min(w + 9L, 24L) - w),
                                         })))))))
           .Build();
        }

        CompilerServices.InferenceType(main);
        var prmg = CompileSession.CreatePassManager("prmg");
        prmg.AddWithName<PrimFuncPass>("unroll").Configure(
          p =>
          {
              p.Add<UnRollLoopSequential>();
          });

        var module = new IR.IRModule(main);
        await prmg.RunAsync(module);
        var post = (TIR.PrimFunction)module.Entry!;
        {
            var getRegion = (int i, ParameterInfo info) =>
            {
                var bufferRegion = (BufferRegion)((Call)post.Body.Fields[i])[info];
                return bufferRegion.Region.AsValueEnumerable().Select(rg =>
                  rg.Start.Evaluate().AsTensor().ToScalar<int>()..rg.Stop.Evaluate().AsTensor().ToScalar<int>())
                  .ToArray();
            };
            int count = 0;
            for (int n = 0; n < 3; n += 3)
            {
                for (int c = 0; c < 16; c += 10)
                {
                    for (int h = 0; h < 24; h += 5)
                    {
                        for (int w = 0; w < 24; w += 9)
                        {
                            var ddrPp = new System.Range[] {
                                        n..System.Math.Min(n + 3, 3),
                                        c..System.Math.Min(c + 10, 16),
                                        h..System.Math.Min(h + 5, 24),
                                        w..System.Math.Min(w + 9, 24),
                            };
                            Assert.True(ddrPp.SequenceEqual(getRegion(count, LoadT.DdrPp)));
                            var glbPp = new System.Range[] {
                                        0..(System.Math.Min(n + 3, 3) - n),
                                        0..(System.Math.Min(c + 10, 16) - c),
                                        0..(System.Math.Min(h + 5, 24) - h),
                                        0..(System.Math.Min(w + 9, 24) - w),
                            };

                            Assert.True(glbPp.SequenceEqual(getRegion(count, LoadT.GlbPp)));
                            count++;
                        }
                    }
                }
            }
        }
    }

    [Fact]
    public async Task TestUnRollLoopSequential3()
    {
        T.CreateBuffer(new TensorType(DataTypes.BFloat16, new[] { 3, 16, 24, 24 }), MemoryLocation.Input, out var ddr_if);
        T.CreateBuffer(new TensorType(DataTypes.BFloat16, new[] { 3, 10, 5, 9 }), MemoryLocation.Data, out var glb_if);

        PrimFunction main;
        {
            main = T.PrimFunc("main", Callable.CPUModuleKind).Body(
             T.Unrolled(out var n, (0, 3, 3)).Body(
               T.Unrolled(out var c, (0, 16, 10)).Body(
                 T.Unrolled(out var h, (0, 24, 5)).Body(
                   T.Unrolled(out var w, (0, 24, 9)).Body(
                     new Call(
                         new LoadT(),
                         new BufferRegion(
                           ddr_if,
                           new TIR.Range[] { (n, Dimension.Min(n + 3L, 3L)),
                                        (c, Dimension.Min(c + 10L, 16L)),
                                        (h, Dimension.Min(h + 5L, 24L)),
                                        (w, Dimension.Min(w + 9L, 24L)),
                                         }),
                         new BufferRegion(
                           glb_if,
                           new TIR.Range[] { (0, Dimension.Min(n + 3L, 3L) - n),
                                        (0, Dimension.Min(c + 10L, 16L) - c),
                                        (0, Dimension.Min(h + 5L, 24L) - h),
                                        (0, Dimension.Min(w + 9L, 24L) - w),
                                         })),
                     T.Unrolled(out var tcu_h, (h, Dimension.Min(h + 5L, 24L), 2L)).Body(
                      new Call(
                         new LoadT(),
                         new BufferRegion(
                           ddr_if,
                           new TIR.Range[] { (n, Dimension.Min(n + 3L, 3L)),
                                        (c, Dimension.Min(c + 10L, 16L)),
                                        (h + tcu_h, Dimension.Min(h + tcu_h + 2L, 24L)),
                                        (w, Dimension.Min(w + 9L, 24L)),
                                         }),
                         new BufferRegion(
                           glb_if,
                           new TIR.Range[] { (0, Dimension.Min(n + 3L, 3L) - n),
                                        (0, Dimension.Min(c + 10L, 16L) - c),
                                        (tcu_h, Dimension.Min(h + tcu_h + 2L, 24L) - h),
                                        (0, Dimension.Min(w + 9L, 24L) - w),
                                         }))))))))
           .Build();
        }

        CompilerServices.InferenceType(main);
        var prmg = CompileSession.CreatePassManager("prmg");
        prmg.AddWithName<PrimFuncPass>("unroll").Configure(
          p =>
          {
              p.Add<UnRollLoopSequential>();
          });

        var module = new IR.IRModule(main);
        await prmg.RunAsync(module);
        var post = (TIR.PrimFunction)module.Entry!;
        {
            var getRegion = (int i, ParameterInfo info) =>
            {
                var bufferRegion = (BufferRegion)((Call)post.Body.Fields[i])[info];
                return bufferRegion.Region.AsValueEnumerable().Select(rg =>
                  rg.Start.Evaluate().AsTensor().ToScalar<int>()..rg.Stop.Evaluate().AsTensor().ToScalar<int>())
                  .ToArray();
            };
            int count = 0;
            for (int n = 0; n < 3; n += 3)
            {
                for (int c = 0; c < 16; c += 10)
                {
                    for (int h = 0; h < 24; h += 5)
                    {
                        for (int w = 0; w < 24; w += 9)
                        {
                            var ddrPp = new System.Range[] {
                                        n..System.Math.Min(n + 3, 3),
                                        c..System.Math.Min(c + 10, 16),
                                        h..System.Math.Min(h + 5, 24),
                                        w..System.Math.Min(w + 9, 24),
                            };
                            Assert.Equal(ddrPp, getRegion(count, LoadT.DdrPp));
                            var glbPp = new System.Range[] {
                                        0..(System.Math.Min(n + 3, 3) - n),
                                        0..(System.Math.Min(c + 10, 16) - c),
                                        0..(System.Math.Min(h + 5, 24) - h),
                                        0..(System.Math.Min(w + 9, 24) - w),
                            };
                            Assert.Equal(glbPp, getRegion(count, LoadT.GlbPp));
                            count++;
                            for (int tcu_h = h; tcu_h < System.Math.Min(h + 5, 24); tcu_h += 2)
                            {
                                var ddrPp2 = new System.Range[] {
                                        n..System.Math.Min(n + 3, 3),
                                        c..System.Math.Min(c + 10, 16),
                                        (h + tcu_h)..System.Math.Min(h + tcu_h + 2, 24),
                                        w..System.Math.Min(w + 9, 24),
                                };
                                Assert.Equal(ddrPp2, getRegion(count, LoadT.DdrPp));
                                var glbPp2 = new System.Range[] {
                                        0..(System.Math.Min(n + 3, 3) - n),
                                        0..(System.Math.Min(c + 10, 16) - c),
                                        tcu_h..(System.Math.Min(h + tcu_h + 2, 24) - h),
                                        0..(System.Math.Min(w + 9, 24) - w),
                                };
                                Assert.Equal(glbPp2, getRegion(count, LoadT.GlbPp));
                                count++;
                            }
                        }
                    }
                }
            }
        }
    }

    [Fact]
    public async Task TestFoldLet()
    {
        var main = T.PrimFunc("main", Callable.CPUModuleKind).Body(// (*i8) -> ()
          T.Unrolled(out var i, (0, 32, 4)).Body(// ()
            T.LetDim(out var a, (Dimension)10L - (Dimension)2L).Body(
              T.LetDim(out var b, (Dimension)10L + (Dimension)2L).Body(
                new Call(new ExtraW(), i + a + b)))))
        .Build();

        CompilerServices.InferenceType(main);

        var pass = new PrimFuncPass { Name = "FoldLet" };
        pass.Add<FoldLet>();
        pass.Add<FoldConstCall>();
        pass.Add<FlattenSequential>();
        var new_func = await pass.RunAsync(main, new());
        Assert.True(new_func.Body[0] is TIR.For for1
                    && for1.Body[0] is Call);
    }

    [Fact]
    public async Task TestFoldLet2()
    {
        var main = T.PrimFunc("main", Callable.CPUModuleKind).Body(// (*i8) -> ()
            T.Let(out var tcu_h_chunk, IR.F.Math.Min(10 + 9L, 32L) - 10L).Body(
              T.Let(out var n_active_tcu, IR.F.Tensors.Cast(IR.F.Math.Ceil(48.0f / IR.F.Tensors.Cast(tcu_h_chunk, DataTypes.Float32)), DataTypes.Int64)).Body(
                T.If(IR.F.Math.Equal(n_active_tcu, 1L)).Then(
                  new Call(new ExtraW(), (Dimension)123))
                .Else(
                  new Call(new ExtraW(), (Dimension)456)))))
        .Build();

        CompilerServices.InferenceType(main);

        var pass = new PrimFuncPass { Name = "FoldLet" };
        pass.Add<FoldLet>();
        pass.Add<FoldConstCall>();
        pass.Add<FlattenSequential>();
        pass.Add<FoldIfThen>();
        var new_func = await pass.RunAsync(main, new());
        Assert.True(new_func.Body[0] is Call { Target: ExtraW } call && call.Arguments[0].Evaluate().AsTensor().ToScalar<int>() == 456);
    }

    [Fact]
    public async Task TestFoldBufferIndex()
    {
        T.CreateBufferVar(new(DataTypes.BFloat16, new[] { 3, 16, 24, 24 }), out var ddr_if);
        T.CreateBufferVar(new(DataTypes.BFloat16, new[] { 3, 16, 24, 24 }), out var ddr_of);
        T.CreateBuffer(new(DataTypes.BFloat16, new[] { 3, 10, 5, 9 }), MemoryLocation.Data, out var glb_if);
        var bufferIndexMap = new Dictionary<Expr, int>() {
          { ddr_if, 2 },
          { ddr_of, 4 },
        };

        PrimFunction main;
        {
            main = T.PrimFunc("main", Callable.CPUModuleKind, ddr_if, ddr_of).Body(
             T.Unrolled(out var n, (0, 3, 3)).Body(
               T.Unrolled(out var c, (0, 16, 10)).Body(
                 T.Unrolled(out var h, (0, 24, 5)).Body(
                   T.Unrolled(out var w, (0, 24, 9)).Body(
                     new Call(new ExtraW(), IR.F.Buffer.BufferIndexOf(ddr_if)),
                     new Call(new ExtraW(), IR.F.Buffer.BufferIndexOf(ddr_of)))))))
           .Build();
        }

        var pass = new PrimFuncPass { Name = "AssginBuffer" };
        pass.Add<UnRollLoopSequential>();
        pass.Add<Substitutor>(BaseExpr? (BaseExpr e) =>
        {
            if (e is Call { } call && call.Arguments[0] is Var physicalBuffer && bufferIndexMap.TryGetValue(physicalBuffer, out var index))
            {
                return (Dimension)index;
            }

            return null;
        });
        pass.Add<FlattenSequential>();
        var post = await pass.RunAsync(main, new());
        {
            var getIndex = (int i, ParameterInfo info) =>
            {
                var index = (DimConst)((Call)post.Body.Fields[i])[info];
                return index.Value;
            };
            int count = 0;
            for (int n = 0; n < 3; n += 3)
            {
                for (int c = 0; c < 16; c += 10)
                {
                    for (int h = 0; h < 24; h += 5)
                    {
                        for (int w = 0; w < 24; w += 9)
                        {
                            Assert.Equal(2, getIndex(count++, ExtraW.Input));
                            Assert.Equal(4, getIndex(count++, ExtraW.Input));
                        }
                    }
                }
            }
        }
    }
}
