export CC=`basename $CC`
export CXX=`basename $CXX`
export LIBRARY_PATH=$PREFIX/lib

pushd ompi && \
    ./configure --prefix=$PREFIX \
                --disable-dependency-tracking \
                --disable-mpi-fortran \
                --disable-wrapper-rpath \
                --disable-wrapper-runpath \
                --with-cuda \
                --with-wrapper-cflags="-I$PREFIX/include" \
                --with-wrapper-cxxflags="-I$PREFIX/include" \
                --with-wrapper-ldflags="-L$PREFIX/lib -Wl,-rpath,$PREFIX/lib" && \
    make -j${CPU_COUNT} all && \
    make install && \
    popd

#--with-sge \
#--with-slrum \
