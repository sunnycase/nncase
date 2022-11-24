using System.Collections;
using System.Diagnostics;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.TIR;
using Nncase.Utilities;
using static Nncase.TestFixture.DumpPathExtractor;

namespace Nncase.TestFixture;

public record OriginValue(IValue Value, string Path)
{
    public OriginTensor[] AsTensors() => Value.AsTensors().Select(t => new OriginTensor(t, Path)).ToArray();

    public OriginTensor AsTensor() => new OriginTensor(Value.AsTensor(), Path);

    public string FileName => System.IO.Path.GetFileName(Path);
}

public record OriginTensor(Tensor Tensor, string Path) : OriginValue(Nncase.Value.FromTensor(Tensor), Path)
{
}

public static class DumpPathExtractor
{
    public static char Separator => '$';
    public static int GetCount(string file) => int.Parse(file.Split(Separator).Head());

    public static string GetOpName(string file) => file.Split(Separator)[1];

    // todo: is param
    public static string GetParamName(string file) => file.Split(Separator).Last();

    public static bool IsResultFile(string file) => file.Count(c => c == Separator) == 1;

    public static bool IsParamFile(string file) => file.Count(c => c == Separator) == 2;

    // used for transformer
    public static bool DynamicMatmulOnlyExtract(string fileName)
    {
        var lower = fileName.ToLower();
        return lower.Contains("mat") && lower.EndsWith("mul");
    }
}

public class TextDataExtractor
{
    // FileNameFormat
    // input: (\d+)*$[a-z]*
    // param: (\d+)*$[a-z]*$[a-z]*
    public int GetDumpFileNum(string filePath)
    {
        var fileName = Path.GetFileName(filePath);
        if (fileName.Contains("out_shape_list"))
        {
            return -1;
        }

        var match = System.Text.RegularExpressions.Regex
            .Match(fileName, @"(\d+)*");
        return int.Parse(match.Groups[0].Value);
    }

    public int FileNumSorter(string x, string y)
    {
        var a = GetDumpFileNum(x);
        var b = GetDumpFileNum(y);
        return a.CompareTo(b);
    }

    public List<string> GetFilesByOrdered(string dir)
    {
        var fs = Directory.GetFiles(dir).ToList();
        fs.Sort(FileNumSorter);
        // remove out shape list
        fs.RemoveAt(0);
        return fs;
    }

    public IEnumerable<IGrouping<string, string>> GetFilesByGroup(string dir)
    {
        return GetFilesByOrdered(dir).GroupBy(file => string.Join(Separator, file.Split(Separator)[..2]));
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="dir"></param>
    /// <returns> dict:num$op_name -> num$op_name$param_name/returns>
    public Dictionary<string, IEnumerable<string>> GetFilesByOrderNum(string dir)
    {
        return GetFilesByGroup(dir)
            .ToDictionary(
            x =>
            {
                var split = x.Key.Split(Separator);
                return $"{split[0].Split(Path.DirectorySeparatorChar).Last()}{Separator}{split[1]}";
            },
            x => x.Select(s => s));
    }

    public OriginValue[] ExtractValues(string dir, Func<string, bool> Extractor)
    {
        var fs = GetFilesByOrdered(dir);
        return fs
            .Filter(filePath => Extractor(Path.GetFileName(filePath)))
            .Select(path => new OriginValue(DataGenerator.FromTextFile(path), path))
            .ToArray();
    }

    public OriginValue[] GetComputeResults(string dir) => ExtractValues(dir, IsResultFile);

    public OriginValue GetComputeResult(string dir, int i)
    {
        var results = ExtractValues(dir, f => IsResultFile(f) && GetDumpFileNum(f) == i);
        Debug.Assert(results.Length != 0);
        return results.Head();
    }

    public OriginValue[] GetParams(string dir, int count) => ExtractValues(dir,
        file => IsParamFile(file) && GetCount(file) == count);

    public OriginValue[] GetValues(string dir)
    {
        return ExtractValues(dir, _ => true);
    }

    public OriginValue[] OpExtract(string dir, string opName)
        => ExtractValues(dir, file => GetOpName(file) == opName);

    public OriginValue[] MatmulExtract(string dir)
    {
        return ExtractValues(dir, DynamicMatmulOnlyExtract).ToArray();
    }
}