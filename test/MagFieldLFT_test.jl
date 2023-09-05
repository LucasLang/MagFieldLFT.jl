using Test

using MagFieldLFT

function test_iscanonical1()
    orbstring = [1,2,5,5,6]    # same orbital occurs twice: not canonical
    return MagFieldLFT.iscanonical(orbstring) == false
end

function test_iscanonical2()
    orbstring = [1,2,5,6]  
    return MagFieldLFT.iscanonical(orbstring) == true
end

function test_iscanonical3()
    orbstring = [2,1,5,6]  
    return MagFieldLFT.iscanonical(orbstring) == false
end

function test_createSDs()
    ref = [[1,2,3], [1,2,4], [1,3,4], [2,3,4]]
    MagFieldLFT.create_SDs(3,2) == ref
end

function test_U_complex2real()
    ref = [1/sqrt(2) 0 0 0 -im/sqrt(2);
           0 -1/sqrt(2) 0 im/sqrt(2) 0;
           0 0 1 0 0;
           0 1/sqrt(2) 0 im/sqrt(2) 0;
           1/sqrt(2) 0 0 0 im/sqrt(2)]
    return MagFieldLFT.U_complex2real(2) ≈ ref
end

@testset "MagFieldLFT.jl" begin
    @test test_iscanonical1()
    @test test_iscanonical2()
    @test test_iscanonical3()
    @test test_createSDs()
    @test test_U_complex2real()
end