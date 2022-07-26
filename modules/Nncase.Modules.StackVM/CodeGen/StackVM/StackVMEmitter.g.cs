// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
/* This file is generated by tools/stackvm_gen/IsaGen at 2022/7/15 16:07:14 +08:00. */

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.Runtime.StackVM;

namespace Nncase.CodeGen.StackVM;

public partial class StackVMEmitter
{
    private readonly BinaryWriter _writer;

    ///<summary>Adds two values and pushes the result onto the evaluation stack.</summary>
    public void Add()
    {
        Write((byte)57);
    }

    ///<summary>Computes the bitwise AND of two values and pushes the result onto the evaluation stack.</summary>
    public void And()
    {
        Write((byte)64);
    }

    ///<summary>Unconditionally transfers control to a target instruction.</summary>
    public void Br(int target)
    {
        Write((byte)91);
        Write(target);
    }

    ///<summary>Inform the debugger that a break point has been tripped.</summary>
    public void Break()
    {
        Write((byte)100);
    }

    ///<summary>Transfers control to a target instruction if value is false, null, or zero.</summary>
    public void BrFalse(int target)
    {
        Write((byte)93);
        Write(target);
    }

    ///<summary>Transfers control to a target instruction if value is true, not null, or non-zero.</summary>
    public void BrTrue(int target)
    {
        Write((byte)92);
        Write(target);
    }

    ///<summary>Call a target method.</summary>
    public void Call(ushort args, int target)
    {
        Write((byte)95);
        Write(args);
        Write(target);
    }

    ///<summary>Compares two values. If they are equal, the integer value 1 (int32) is pushed onto the evaluation stack; otherwise 0 (int32) is pushed onto the evaluation stack.</summary>
    public void Ceq()
    {
        Write((byte)75);
    }

    ///<summary>Compares two values. If the first value is greater than or equal to the second, the integer value 1 (int32) is pushed onto the evaluation stack; otherwise 0 (int32) is pushed onto the evaluation stack.</summary>
    public void Cge()
    {
        Write((byte)76);
    }

    ///<summary>Compares the unsigned or unordered values value1 and value2. If value1 is greater than or equal to value2, then the integer value 1 (int32) is pushed onto the evaluation stack; otherwise 0 (int32) is pushed onto the evaluation stack.</summary>
    public void CgeU()
    {
        Write((byte)77);
    }

    ///<summary>Compares two values. If the first value is greater than the second, the integer value 1 (int32) is pushed onto the evaluation stack; otherwise 0 (int32) is pushed onto the evaluation stack.</summary>
    public void Cgt()
    {
        Write((byte)78);
    }

    ///<summary>Compares the unsigned or unordered values value1 and value2. If value1 is greater than value2, then the integer value 1 (int32) is pushed onto the evaluation stack; otherwise 0 (int32) is pushed onto the evaluation stack.</summary>
    public void CgtU()
    {
        Write((byte)79);
    }

    ///<summary>Compares two values. If the first value is less than or equal to the second, the integer value 1 (int32) is pushed onto the evaluation stack; otherwise 0 (int32) is pushed onto the evaluation stack.</summary>
    public void Cle()
    {
        Write((byte)73);
    }

    ///<summary>Compares the unsigned or unordered values value1 and value2. If value1 is less than or equal to value2, then the integer value 1 (int32) is pushed onto the evaluation stack; otherwise 0 (int32) is pushed onto the evaluation stack.</summary>
    public void CleU()
    {
        Write((byte)74);
    }

    ///<summary>Compares two values. If the first value is less than the second, the integer value 1 (int32) is pushed onto the evaluation stack; otherwise 0 (int32) is pushed onto the evaluation stack.</summary>
    public void Clt()
    {
        Write((byte)71);
    }

    ///<summary>Compares the unsigned or unordered values value1 and value2. If value1 is less than value2, then the integer value 1 (int32) is pushed onto the evaluation stack; otherwise 0 (int32) is pushed onto the evaluation stack.</summary>
    public void CltU()
    {
        Write((byte)72);
    }

    ///<summary>Compares two values. If the first value is not equal to the second, the integer value 1 (int32) is pushed onto the evaluation stack; otherwise 0 (int32) is pushed onto the evaluation stack.</summary>
    public void Cne()
    {
        Write((byte)80);
    }

    ///<summary>Converts the value on top of the evaluation stack to bfloat16.</summary>
    public void ConvBR2()
    {
        Write((byte)89);
    }

    ///<summary>Converts the value on top of the evaluation stack to native int, and extends it to int32.</summary>
    public void ConvI()
    {
        Write((byte)84);
    }

    ///<summary>Converts the value on top of the evaluation stack to int8, and extends it to int32.</summary>
    public void ConvI1()
    {
        Write((byte)81);
    }

    ///<summary>Converts the value on top of the evaluation stack to int16, and extends it to int32.</summary>
    public void ConvI2()
    {
        Write((byte)82);
    }

    ///<summary>Converts the value on top of the evaluation stack to int32, and extends it to int32.</summary>
    public void ConvI4()
    {
        Write((byte)83);
    }

    ///<summary>Converts the value on top of the evaluation stack to float32.</summary>
    public void ConvR4()
    {
        Write((byte)90);
    }

    ///<summary>Converts the value on top of the evaluation stack to unsigned native int, and extends it to int32.</summary>
    public void ConvU()
    {
        Write((byte)88);
    }

    ///<summary>Converts the value on top of the evaluation stack to unsigned int8, and extends it to int32.</summary>
    public void ConvU1()
    {
        Write((byte)85);
    }

    ///<summary>Converts the value on top of the evaluation stack to unsigned int16, and extends it to int32.</summary>
    public void ConvU2()
    {
        Write((byte)86);
    }

    ///<summary>Converts the value on top of the evaluation stack to unsigned int32, and extends it to int32.</summary>
    public void ConvU4()
    {
        Write((byte)87);
    }

    ///<summary>Custom Call an User customed method.</summary>
    public void CusCall(string registered_name, Byte[] fields_span, ushort args)
    {
        Write((byte)98);
        Write(registered_name);
        Write(fields_span);
        Write(args);
    }

    ///<summary>Divides two values and pushes the result as a floating-point (type F) or quotient (type int32) onto the evaluation stack.</summary>
    public void Div()
    {
        Write((byte)60);
    }

    ///<summary>Divides two unsigned integer values and pushes the result (int32) onto the evaluation stack.</summary>
    public void DivU()
    {
        Write((byte)61);
    }

    ///<summary>Duplicate the top item of stack.</summary>
    public void Dup()
    {
        Write((byte)46);
    }

    ///<summary>Call an environment method.</summary>
    public void ECall(ushort args)
    {
        Write((byte)96);
        Write(args);
    }

    ///<summary>Call an external method.</summary>
    public void ExtCall(ushort args, bool isPrimFunc)
    {
        Write((byte)97);
        Write(args);
        Write(isPrimFunc);
    }

    ///<summary>Load an argument to stack.</summary>
    public void Ldarg(ushort index)
    {
        Write((byte)39);
        Write(index);
    }

    ///<summary>Load an argument with index of 0 to stack.</summary>
    public void Ldarg0()
    {
        Write((byte)40);
    }

    ///<summary>Load an argument with index of 1 to stack.</summary>
    public void Ldarg1()
    {
        Write((byte)41);
    }

    ///<summary>Load an argument with index of 2 to stack.</summary>
    public void Ldarg2()
    {
        Write((byte)42);
    }

    ///<summary>Load an argument with index of 1 to stack.</summary>
    public void Ldarg3()
    {
        Write((byte)43);
    }

    ///<summary>Load an argument with index of 4 to stack.</summary>
    public void Ldarg4()
    {
        Write((byte)44);
    }

    ///<summary>Load an argument with index of 5 to stack.</summary>
    public void Ldarg5()
    {
        Write((byte)45);
    }

    ///<summary>Load immedidate I4 to stack.</summary>
    public void LdcI4(int imm)
    {
        Write((byte)2);
        Write(imm);
    }

    ///<summary>Load immedidate 0 as I4 to stack.</summary>
    public void LdcI4_0()
    {
        Write((byte)3);
    }

    ///<summary>Load immedidate 1 as I4 to stack.</summary>
    public void LdcI4_1()
    {
        Write((byte)4);
    }

    ///<summary>Load immedidate R4 to stack.</summary>
    public void LdcR4(float imm)
    {
        Write((byte)5);
        Write(imm);
    }

    ///<summary>Load a datatype to stack.</summary>
    public void LdDataType()
    {
        Write((byte)54);
    }

    ///<summary>Load an array element of BR2 to stack.</summary>
    public void LdelemBR2()
    {
        Write((byte)31);
    }

    ///<summary>Load an array element of I to stack.</summary>
    public void LdelemI()
    {
        Write((byte)26);
    }

    ///<summary>Load an array element of I1 to stack.</summary>
    public void LdelemI1()
    {
        Write((byte)23);
    }

    ///<summary>Load an array element of I2 to stack.</summary>
    public void LdelemI2()
    {
        Write((byte)24);
    }

    ///<summary>Load an array element of I4 to stack.</summary>
    public void LdelemI4()
    {
        Write((byte)25);
    }

    ///<summary>Load an array element of R4 to stack.</summary>
    public void LdelemR4()
    {
        Write((byte)32);
    }

    ///<summary>Load an array element of U to stack.</summary>
    public void LdelemU()
    {
        Write((byte)30);
    }

    ///<summary>Load an array element of U1 to stack.</summary>
    public void LdelemU1()
    {
        Write((byte)27);
    }

    ///<summary>Load an array element of U2 to stack.</summary>
    public void LdelemU2()
    {
        Write((byte)28);
    }

    ///<summary>Load an array element of U4 to stack.</summary>
    public void LdelemU4()
    {
        Write((byte)29);
    }

    ///<summary>Load indirect BR2 to stack.</summary>
    public void LdindBR2()
    {
        Write((byte)14);
    }

    ///<summary>Load indirect I to stack.</summary>
    public void LdindI()
    {
        Write((byte)9);
    }

    ///<summary>Load indirect I1 to stack.</summary>
    public void LdindI1()
    {
        Write((byte)6);
    }

    ///<summary>Load indirect I2 to stack.</summary>
    public void LdindI2()
    {
        Write((byte)7);
    }

    ///<summary>Load indirect I4 to stack.</summary>
    public void LdindI4()
    {
        Write((byte)8);
    }

    ///<summary>Load indirect R4 to stack.</summary>
    public void LdindR4()
    {
        Write((byte)15);
    }

    ///<summary>Load indirect U to stack.</summary>
    public void LdindU()
    {
        Write((byte)13);
    }

    ///<summary>Load indirect U1 to stack.</summary>
    public void LdindU1()
    {
        Write((byte)10);
    }

    ///<summary>Load indirect U2 to stack.</summary>
    public void LdindU2()
    {
        Write((byte)11);
    }

    ///<summary>Load indirect U4 to stack.</summary>
    public void LdindU4()
    {
        Write((byte)12);
    }

    ///<summary>Load a local to stack.</summary>
    public void Ldlocal(ushort index)
    {
        Write((byte)48);
        Write(index);
    }

    ///<summary>Load immedidate nullptr as I to stack.</summary>
    public void LdNull()
    {
        Write((byte)1);
    }

    ///<summary>Load a shape to stack.</summary>
    public void LdShape()
    {
        Write((byte)50);
    }

    ///<summary>Load a strides to stack.</summary>
    public void LdStrides()
    {
        Write((byte)51);
    }

    ///<summary>Load a tensor to stack.</summary>
    public void LdTensor()
    {
        Write((byte)55);
    }

    ///<summary>Load a tuple to stack.</summary>
    public void LdTuple()
    {
        Write((byte)53);
    }

    ///<summary>Load an element of tuple to stack.</summary>
    public void LdTupleElem()
    {
        Write((byte)52);
    }

    ///<summary>Load a global pointer with offset to stack.</summary>
    public void LeaGP(byte gpid, int offset)
    {
        Write((byte)22);
        Write(gpid);
        Write(offset);
    }

    ///<summary>Multiplies two values and pushes the result on the evaluation stack.</summary>
    public void Mul()
    {
        Write((byte)59);
    }

    ///<summary>Negates a value and pushes the result onto the evaluation stack.</summary>
    public void Neg()
    {
        Write((byte)56);
    }

    ///<summary>No operation.</summary>
    public void Nop()
    {
        Write((byte)0);
    }

    ///<summary>Computes the bitwise complement of the integer value on top of the stack and pushes the result onto the evaluation stack as the same type.</summary>
    public void Not()
    {
        Write((byte)67);
    }

    ///<summary>Compute the bitwise complement of the two integer values on top of the stack and pushes the result onto the evaluation stack.</summary>
    public void Or()
    {
        Write((byte)65);
    }

    ///<summary>Pop the top item of stack.</summary>
    public void Pop()
    {
        Write((byte)47);
    }

    ///<summary>Divides two values and pushes the remainder onto the evaluation stack.</summary>
    public void Rem()
    {
        Write((byte)62);
    }

    ///<summary>Divides two unsigned values and pushes the remainder onto the evaluation stack.</summary>
    public void RemU()
    {
        Write((byte)63);
    }

    ///<summary>Return.</summary>
    public void Ret()
    {
        Write((byte)94);
    }

    ///<summary>Shifts an integer value to the left (in zeroes) by a specified number of bits, pushing the result onto the evaluation stack.</summary>
    public void Shl()
    {
        Write((byte)68);
    }

    ///<summary>Shifts an integer value (in sign) to the right by a specified number of bits, pushing the result onto the evaluation stack.</summary>
    public void Shr()
    {
        Write((byte)69);
    }

    ///<summary>Shifts an unsigned integer value (in zeroes) to the right by a specified number of bits, pushing the result onto the evaluation stack.</summary>
    public void ShrU()
    {
        Write((byte)70);
    }

    ///<summary>Store an array element of BR2 from stack.</summary>
    public void StelemBR2()
    {
        Write((byte)37);
    }

    ///<summary>Store an array element of I from stack.</summary>
    public void StelemI()
    {
        Write((byte)36);
    }

    ///<summary>Store an array element of I1 from stack.</summary>
    public void StelemI1()
    {
        Write((byte)33);
    }

    ///<summary>Store an array element of I2 from stack.</summary>
    public void StelemI2()
    {
        Write((byte)34);
    }

    ///<summary>Store an array element of I4 from stack.</summary>
    public void StelemI4()
    {
        Write((byte)35);
    }

    ///<summary>Store an array element of R4 from stack.</summary>
    public void StelemR4()
    {
        Write((byte)38);
    }

    ///<summary>Store indirect BR2 from stack.</summary>
    public void StindBR2()
    {
        Write((byte)20);
    }

    ///<summary>Store indirect I from stack.</summary>
    public void StindI()
    {
        Write((byte)19);
    }

    ///<summary>Store indirect I1 from stack.</summary>
    public void StindI1()
    {
        Write((byte)16);
    }

    ///<summary>Store indirect I2 from stack.</summary>
    public void StindI2()
    {
        Write((byte)17);
    }

    ///<summary>Store indirect I4 from stack.</summary>
    public void StindI4()
    {
        Write((byte)18);
    }

    ///<summary>Store indirect R4 from stack.</summary>
    public void StindR4()
    {
        Write((byte)21);
    }

    ///<summary>Store a local from stack.</summary>
    public void Stlocal(ushort index)
    {
        Write((byte)49);
        Write(index);
    }

    ///<summary>Subtracts one value from another and pushes the result onto the evaluation stack.</summary>
    public void Sub()
    {
        Write((byte)58);
    }

    ///<summary>Throw a error code currently on the evaluation stack.</summary>
    public void Throw()
    {
        Write((byte)99);
    }

    ///<summary>Computes the bitwise XOR of the top two values on the evaluation stack, pushing the result onto the evaluation stack.</summary>
    public void Xor()
    {
        Write((byte)66);
    }

    public partial class TensorEmitter
    {
        private readonly StackVMEmitter _emitter;

        ///<summary>.</summary>
        public void BatchNormalization()
        {
            _emitter.Write((byte)101);
            _emitter.Write((ushort)0);
        }

        ///<summary>.</summary>
        public void BatchToSpace()
        {
            _emitter.Write((byte)101);
            _emitter.Write((ushort)1);
        }

        ///<summary>.</summary>
        public void Binary(BinaryOp binaryOp)
        {
            _emitter.Write((byte)101);
            _emitter.Write((ushort)2);
            _emitter.Write((byte)binaryOp);
        }

        ///<summary>.</summary>
        public void Broadcast()
        {
            _emitter.Write((byte)101);
            _emitter.Write((ushort)3);
        }

        ///<summary>.</summary>
        public void Cast(DataType newType)
        {
            _emitter.Write((byte)101);
            _emitter.Write((ushort)4);
            _emitter.Write(newType);
        }

        ///<summary>.</summary>
        public void Celu()
        {
            _emitter.Write((byte)101);
            _emitter.Write((ushort)5);
        }

        ///<summary>.</summary>
        public void Clamp()
        {
            _emitter.Write((byte)101);
            _emitter.Write((ushort)6);
        }

        ///<summary>.</summary>
        public void Compare(CompareOp compareOp)
        {
            _emitter.Write((byte)101);
            _emitter.Write((ushort)7);
            _emitter.Write((byte)compareOp);
        }

        ///<summary>.</summary>
        public void Concat()
        {
            _emitter.Write((byte)101);
            _emitter.Write((ushort)8);
        }

        ///<summary>.</summary>
        public void ConstantOfShape()
        {
            _emitter.Write((byte)101);
            _emitter.Write((ushort)9);
        }

        ///<summary>.</summary>
        public void Conv2D(PadMode padMode)
        {
            _emitter.Write((byte)101);
            _emitter.Write((ushort)10);
            _emitter.Write((byte)padMode);
        }

        ///<summary>.</summary>
        public void Conv2DTranspose(PadMode padMode)
        {
            _emitter.Write((byte)101);
            _emitter.Write((ushort)11);
            _emitter.Write((byte)padMode);
        }

        ///<summary>.</summary>
        public void CumSum()
        {
            _emitter.Write((byte)101);
            _emitter.Write((ushort)12);
        }

        ///<summary>.</summary>
        public void Dequantize(DataType targetType)
        {
            _emitter.Write((byte)101);
            _emitter.Write((ushort)13);
            _emitter.Write(targetType);
        }

        ///<summary>.</summary>
        public void Elu()
        {
            _emitter.Write((byte)101);
            _emitter.Write((ushort)14);
        }

        ///<summary>.</summary>
        public void Expand()
        {
            _emitter.Write((byte)101);
            _emitter.Write((ushort)15);
        }

        ///<summary>.</summary>
        public void FakeDequantize(DataType targetType)
        {
            _emitter.Write((byte)101);
            _emitter.Write((ushort)16);
            _emitter.Write(targetType);
        }

        ///<summary>.</summary>
        public void FakeQuantize(DataType targetType)
        {
            _emitter.Write((byte)101);
            _emitter.Write((ushort)17);
            _emitter.Write(targetType);
        }

        ///<summary>.</summary>
        public void Flatten()
        {
            _emitter.Write((byte)101);
            _emitter.Write((ushort)18);
        }

        ///<summary>.</summary>
        public void Gather()
        {
            _emitter.Write((byte)101);
            _emitter.Write((ushort)19);
        }

        ///<summary>.</summary>
        public void GatherND()
        {
            _emitter.Write((byte)101);
            _emitter.Write((ushort)20);
        }

        ///<summary>.</summary>
        public void GetItem()
        {
            _emitter.Write((byte)101);
            _emitter.Write((ushort)21);
        }

        ///<summary>.</summary>
        public void Hardmax()
        {
            _emitter.Write((byte)101);
            _emitter.Write((ushort)22);
        }

        ///<summary>.</summary>
        public void HardSigmoid()
        {
            _emitter.Write((byte)101);
            _emitter.Write((ushort)23);
        }

        ///<summary>.</summary>
        public void HardSwish()
        {
            _emitter.Write((byte)101);
            _emitter.Write((ushort)24);
        }

        ///<summary>.</summary>
        public void InstanceNormalization()
        {
            _emitter.Write((byte)101);
            _emitter.Write((ushort)25);
        }

        ///<summary>.</summary>
        public void L2Normalization()
        {
            _emitter.Write((byte)101);
            _emitter.Write((ushort)26);
        }

        ///<summary>.</summary>
        public void LeakyRelu()
        {
            _emitter.Write((byte)101);
            _emitter.Write((ushort)27);
        }

        ///<summary>.</summary>
        public void LogSoftmax()
        {
            _emitter.Write((byte)101);
            _emitter.Write((ushort)28);
        }

        ///<summary>.</summary>
        public void LpNormalization()
        {
            _emitter.Write((byte)101);
            _emitter.Write((ushort)29);
        }

        ///<summary>.</summary>
        public void LRN()
        {
            _emitter.Write((byte)101);
            _emitter.Write((ushort)30);
        }

        ///<summary>.</summary>
        public void LSTM(LSTMDirection direction, LSTMLayout layout, string[] activations)
        {
            _emitter.Write((byte)101);
            _emitter.Write((ushort)31);
            _emitter.Write((int)direction);
            _emitter.Write((int)layout);
            _emitter.Write(activations);
        }

        ///<summary>.</summary>
        public void MatMul()
        {
            _emitter.Write((byte)101);
            _emitter.Write((ushort)32);
        }

        ///<summary>.</summary>
        public void Normal(DataType type)
        {
            _emitter.Write((byte)101);
            _emitter.Write((ushort)33);
            _emitter.Write(type);
        }

        ///<summary>.</summary>
        public void NormalLike(DataType type)
        {
            _emitter.Write((byte)101);
            _emitter.Write((ushort)34);
            _emitter.Write(type);
        }

        ///<summary>.</summary>
        public void OneHot(OneHotMode oneHotMode)
        {
            _emitter.Write((byte)101);
            _emitter.Write((ushort)35);
            _emitter.Write((byte)oneHotMode);
        }

        ///<summary>.</summary>
        public void Pad(PadMode padMode)
        {
            _emitter.Write((byte)101);
            _emitter.Write((ushort)36);
            _emitter.Write((byte)padMode);
        }

        ///<summary>.</summary>
        public void PRelu()
        {
            _emitter.Write((byte)101);
            _emitter.Write((ushort)37);
        }

        ///<summary>.</summary>
        public void Prod()
        {
            _emitter.Write((byte)101);
            _emitter.Write((ushort)38);
        }

        ///<summary>.</summary>
        public void Quantize(DataType targetType)
        {
            _emitter.Write((byte)101);
            _emitter.Write((ushort)39);
            _emitter.Write(targetType);
        }

        ///<summary>.</summary>
        public void QuantParamOf(QuantMode quantMode)
        {
            _emitter.Write((byte)101);
            _emitter.Write((ushort)40);
            _emitter.Write((int)quantMode);
        }

        ///<summary>.</summary>
        public void Range()
        {
            _emitter.Write((byte)101);
            _emitter.Write((ushort)41);
        }

        ///<summary>.</summary>
        public void RangeOf()
        {
            _emitter.Write((byte)101);
            _emitter.Write((ushort)42);
        }

        ///<summary>.</summary>
        public void Reduce(ReduceOp reduceOp)
        {
            _emitter.Write((byte)101);
            _emitter.Write((ushort)43);
            _emitter.Write((byte)reduceOp);
        }

        ///<summary>.</summary>
        public void ReduceArg(ReduceArgOp reduceArgOp)
        {
            _emitter.Write((byte)101);
            _emitter.Write((ushort)44);
            _emitter.Write((byte)reduceArgOp);
        }

        ///<summary>.</summary>
        public void ReduceWindow2D(ReduceOp reduceOp)
        {
            _emitter.Write((byte)101);
            _emitter.Write((ushort)45);
            _emitter.Write((byte)reduceOp);
        }

        ///<summary>.</summary>
        public void Relu()
        {
            _emitter.Write((byte)101);
            _emitter.Write((ushort)46);
        }

        ///<summary>.</summary>
        public void Relu6()
        {
            _emitter.Write((byte)101);
            _emitter.Write((ushort)47);
        }

        ///<summary>.</summary>
        public void Require(string message)
        {
            _emitter.Write((byte)101);
            _emitter.Write((ushort)48);
            _emitter.Write(message);
        }

        ///<summary>.</summary>
        public void Reshape()
        {
            _emitter.Write((byte)101);
            _emitter.Write((ushort)49);
        }

        ///<summary>.</summary>
        public void ResizeImage(ImageResizeMode resizeMode, ImageResizeTransformationMode transformationMode, ImageResizeNearestMode nearestMode, bool isTFResize)
        {
            _emitter.Write((byte)101);
            _emitter.Write((ushort)50);
            _emitter.Write((byte)resizeMode);
            _emitter.Write((int)transformationMode);
            _emitter.Write((int)nearestMode);
            _emitter.Write(isTFResize);
        }

        ///<summary>.</summary>
        public void ReverseSequence()
        {
            _emitter.Write((byte)101);
            _emitter.Write((ushort)51);
        }

        ///<summary>.</summary>
        public void Select()
        {
            _emitter.Write((byte)101);
            _emitter.Write((ushort)52);
        }

        ///<summary>.</summary>
        public void Selu()
        {
            _emitter.Write((byte)101);
            _emitter.Write((ushort)53);
        }

        ///<summary>.</summary>
        public void ShapeOf()
        {
            _emitter.Write((byte)101);
            _emitter.Write((ushort)54);
        }

        ///<summary>.</summary>
        public void Sigmoid()
        {
            _emitter.Write((byte)101);
            _emitter.Write((ushort)55);
        }

        ///<summary>.</summary>
        public void SizeOf()
        {
            _emitter.Write((byte)101);
            _emitter.Write((ushort)56);
        }

        ///<summary>.</summary>
        public void Slice()
        {
            _emitter.Write((byte)101);
            _emitter.Write((ushort)57);
        }

        ///<summary>.</summary>
        public void Softmax()
        {
            _emitter.Write((byte)101);
            _emitter.Write((ushort)58);
        }

        ///<summary>.</summary>
        public void Softplus()
        {
            _emitter.Write((byte)101);
            _emitter.Write((ushort)59);
        }

        ///<summary>.</summary>
        public void Softsign()
        {
            _emitter.Write((byte)101);
            _emitter.Write((ushort)60);
        }

        ///<summary>.</summary>
        public void SpaceToBatch()
        {
            _emitter.Write((byte)101);
            _emitter.Write((ushort)61);
        }

        ///<summary>.</summary>
        public void Split()
        {
            _emitter.Write((byte)101);
            _emitter.Write((ushort)62);
        }

        ///<summary>.</summary>
        public void Squeeze()
        {
            _emitter.Write((byte)101);
            _emitter.Write((ushort)63);
        }

        ///<summary>.</summary>
        public void Stack()
        {
            _emitter.Write((byte)101);
            _emitter.Write((ushort)64);
        }

        ///<summary>.</summary>
        public void Tile()
        {
            _emitter.Write((byte)101);
            _emitter.Write((ushort)65);
        }

        ///<summary>.</summary>
        public void Transpose()
        {
            _emitter.Write((byte)101);
            _emitter.Write((ushort)66);
        }

        ///<summary>.</summary>
        public void Unary(UnaryOp unaryOp)
        {
            _emitter.Write((byte)101);
            _emitter.Write((ushort)67);
            _emitter.Write((byte)unaryOp);
        }

        ///<summary>.</summary>
        public void Uniform(DataType type)
        {
            _emitter.Write((byte)101);
            _emitter.Write((ushort)68);
            _emitter.Write(type);
        }

        ///<summary>.</summary>
        public void UniformLike(DataType type)
        {
            _emitter.Write((byte)101);
            _emitter.Write((ushort)69);
            _emitter.Write(type);
        }

        ///<summary>.</summary>
        public void Unsqueeze()
        {
            _emitter.Write((byte)101);
            _emitter.Write((ushort)70);
        }

        ///<summary>.</summary>
        public void Where()
        {
            _emitter.Write((byte)101);
            _emitter.Write((ushort)71);
        }
    }
}
