using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Tuple = Nncase.IR.Tuple;

namespace Nncase;


public abstract class BaseImporter
{
    protected SortedSet<string> _opsInModel = new SortedSet<string>();
    protected SortedSet<string> _unsupportedOp = new SortedSet<string>();
    
    public IRModule Import()
    {
        var inputs = CreateInputs().ToArray();
        ConvertOp();
        SupportedCheck(this.GetType().Name.Split("Importer")[0]);
        var outputs = CreateOutputs();
        DumpOpsInModel();
        return CreateModule(inputs.ToArray(), outputs);
    }

    public void AddToOutputs<TKey, TNode>(Dictionary<TKey, Expr> outTensors, TKey[] opOutputs, TNode output)
    {
        var outLength = opOutputs.Length;
        if (output is Expr expr)
        {
            if (opOutputs.Length == 1)
            {
                outTensors.Add(opOutputs[0], expr);
            }
            else
            {
                for (int i = 0; i < outLength; i++)
                {
                    outTensors.Add(opOutputs[i], IR.F.Tensors.GetItem(expr, i));
                }
            }
        }
        else if (output is IReadOnlyList<Expr> exprs)
        {
            Debug.Assert(outLength == exprs.Count, $"Op outputs length should be {outLength}.");
            for (int i = 0; i < outLength; i++)
            {
                outTensors.Add(opOutputs[i], exprs[i]);
            }
        }
        else
        {
            throw new InvalidOperationException("Visit result is not expression(s).");
        }
    }
    public void DumpOpsInModel()
    {
        // todo: fix this
        using (var sr = new StreamWriter("/home/homura/Code/nncase/tests/importer/onnx_/basic/ops.txt"))
        {
            foreach (var op in _opsInModel)
            {
                sr.WriteLine(op);
            }
        }
    }

    public abstract IEnumerable<Var> CreateInputs();
    
    public abstract void ConvertOp();
    
    public abstract Expr CreateOutputs();

    private IRModule CreateModule(Var[] inputs, Expr body)
    {
        var mainFunc = new Function("main", body, inputs);
        var module = new IRModule();
        module.Add(mainFunc);
        module.Entry = mainFunc;
        return module;
    }
    
    protected Expr UnSupportedOp(string opType)
    {
        _unsupportedOp.Append(opType);
        return None.Default;
    }

    protected void SupportedCheck(string name)
    {
        if (_unsupportedOp.Count > 0)
        {
            throw new NotSupportedException(
                $"Not Supported {name} op {_unsupportedOp.Aggregate("", (s, s1) => s + "\n" + s1)}");
        }
    }
}