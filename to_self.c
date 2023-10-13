/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * Copyright (c) 2004-2019 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "mpi.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

#include <sys/mman.h>
#include <zlib.h>

#define MIN_LENGTH 1024
#define MAX_LENGTH (1024 * 1024)

static int cycles = 10;
static int trials = 10;
static int warmups = 1;

static void print_result(int length, int trials, double *timers)
{
    double bandwidth, clock_prec, temp;
    double min_time, max_time, average, std_dev = 0.0;
    double ordered[trials];
    int t, pos, quartile_start, quartile_end;

    for (t = 0; t < trials; ordered[t] = timers[t], t++)
        ;
    for (t = 0; t < trials - 1; t++) {
        temp = ordered[t];
        pos = t;
        for (int i = t + 1; i < trials; i++) {
            if (temp > ordered[i]) {
                temp = ordered[i];
                pos = i;
            }
        }
        if (pos != t) {
            temp = ordered[t];
            ordered[t] = ordered[pos];
            ordered[pos] = temp;
        }
    }
    quartile_start = trials - (3 * trials) / 4;
    quartile_end = trials - (1 * trials) / 4;
    clock_prec = MPI_Wtick();
    min_time = ordered[quartile_start];
    max_time = ordered[quartile_start];
    average = ordered[quartile_start];
    for (t = quartile_start + 1; t < quartile_end; t++) {
        if (min_time > ordered[t])
            min_time = ordered[t];
        if (max_time < ordered[t])
            max_time = ordered[t];
        average += ordered[t];
    }
    average /= (quartile_end - quartile_start);
    for (t = quartile_start; t < quartile_end; t++) {
        std_dev += (ordered[t] - average) * (ordered[t] - average);
    }
    std_dev = sqrt(std_dev / (quartile_end - quartile_start));

    bandwidth = (length * clock_prec) / (1024.0 * 1024.0) / (average * clock_prec);
    printf("%8d\t%15g\t%10.4f MB/s [min %10g max %10g std %2.2f%%]\n", length, average, bandwidth,
           min_time, max_time, (100.0 * std_dev) / average);
}

static int pack(int cycles, MPI_Datatype sdt, int scount, void *sbuf, void *packed_buf)
{
    int position, myself, c, t, outsize;
    double timers[trials];

    MPI_Type_size(sdt, &outsize);
    outsize *= scount;

    MPI_Comm_rank(MPI_COMM_WORLD, &myself);

    for (t = 0; t < warmups; t++) {
	    for (c = 0; c < cycles; c++) {
		    position = 0;
		    MPI_Pack(sbuf, scount, sdt, packed_buf, outsize, &position, MPI_COMM_WORLD);
	    }
    }

    for (t = 0; t < trials; t++) {
	    timers[t] = MPI_Wtime();
	    for (c = 0; c < cycles; c++) {
		    position = 0;
		    MPI_Pack(sbuf, scount, sdt, packed_buf, outsize, &position, MPI_COMM_WORLD);
	    }
	    timers[t] = (MPI_Wtime() - timers[t]) / cycles;
    }

    print_result(outsize, trials, timers);

    return 0;
}

static int unpack(int cycles, void *packed_buf, MPI_Datatype rdt, int rcount, void *rbuf)
{
    int position, myself, c, t, insize;
    double timers[trials];

    MPI_Type_size(rdt, &insize);
    insize *= rcount;

    MPI_Comm_rank(MPI_COMM_WORLD, &myself);

    for (t = 0; t < warmups; t++) {
        for (c = 0; c < cycles; c++) {
            position = 0;
            MPI_Unpack(packed_buf, insize, &position, rbuf, rcount, rdt, MPI_COMM_WORLD);
        }
    }

    for (t = 0; t < trials; t++) {
        timers[t] = MPI_Wtime();
        for (c = 0; c < cycles; c++) {
            position = 0;
            MPI_Unpack(packed_buf, insize, &position, rbuf, rcount, rdt, MPI_COMM_WORLD);
        }
        timers[t] = (MPI_Wtime() - timers[t]) / cycles;
    }
    print_result(insize, trials, timers);
    return 0;
}

//#define COMPRESSION_LEVEL Z_DEFAULT_COMPRESSION
#define COMPRESSION_LEVEL 9

static int compress_buffer(const unsigned char *input_buffer, size_t input_size, unsigned char **output_buffer, size_t *output_size) {

	z_stream strm;
	int ret, rank;

	strm.zalloc = Z_NULL;
	strm.zfree = Z_NULL;
	strm.opaque = Z_NULL;
	strm.total_in = strm.avail_in = (uInt)input_size;
	strm.next_in = (Bytef *)input_buffer;
	strm.total_out = strm.avail_out = (uInt)*output_size;
	strm.next_out = (Bytef *)*output_buffer;

	ret = deflateInit(&strm, COMPRESSION_LEVEL);
	if (ret != Z_OK) {
		printf("not Z_OK\n");
		exit(0);
		return ret;
	}

	ret = deflate(&strm, Z_FINISH);
	if (ret != Z_STREAM_END) {
		printf("Z_STREAM_END %d\n", Z_STREAM_END);
		deflateEnd(&strm);
		printf("compress ret %d Z_STREAM_ERROR %d Z_BUF_ERROR %d\n", 
				ret,
				Z_STREAM_ERROR,
				Z_BUF_ERROR);
		exit(0);
		return ret;
	}

	*output_size = strm.total_out;
	deflateEnd(&strm);
	return Z_OK;
}

static int decompress_buffer(const unsigned char *input_buffer, size_t input_size, unsigned char **output_buffer, size_t *output_size) {
	z_stream strm;
	int ret;

	size_t comp_size = *output_size * 2;
	*output_size = 0UL;

	strm.zalloc = Z_NULL;
	strm.zfree = Z_NULL;
	strm.opaque = Z_NULL;
	strm.avail_in = (uInt)input_size;
	strm.next_in = (Bytef *)input_buffer;
	strm.avail_out = (uInt)comp_size;
	strm.next_out = (Bytef *)*output_buffer;

	ret = inflateInit(&strm);
	if (ret != Z_OK) {
		printf("decompress not Z_OK\n");
		return ret;
	}

	ret = inflate(&strm, Z_FINISH);
	if (ret != Z_STREAM_END) {
		printf("decompress ret %d Z_STREAM_ERROR %d Z_BUF_ERROR %d\n",
                                ret,
                                Z_STREAM_ERROR,
                                Z_BUF_ERROR);
		inflateEnd(&strm);
		return ret;
	}

	*output_size = strm.total_out;
	inflateEnd(&strm);
	return Z_OK;
}

static int send_rank_orig( int cycles, MPI_Datatype sddt, void *sbuf, void* rbuf )
{
	int outsize, do_size, done;
	double timers[trials];

	MPI_Type_size(sddt, &outsize);

	int c = 0;
	for( int t = 0; t < trials; t++ ){
	//	if( t == 0 )
	//		printf("reference_buffer_size %zu ", outsize);
		
		timers[t] = MPI_Wtime();
		for( c = 0; c < cycles; c++ ){
			MPI_Send(sbuf, 1, sddt, 1, t*4 + c, MPI_COMM_WORLD);
			MPI_Recv( rbuf, 1, sddt, 1, t*4+c+1, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
		}
		timers[t] = (MPI_Wtime() - timers[t]) / cycles;
	}

	print_result(outsize, trials, timers);

	return 0;
}

static int recv_rank_orig( int cycles, MPI_Datatype sddt, void *sbuf, void* rbuf )
{
	int do_size;
	int result;

	MPI_Type_size( sddt, &do_size );
	
	int done = 1;
	int c = 0;
	for( int t = 0; t < trials; t++ ){
		// recv and decompress
		for( c = 0; c < cycles; c++ ){
			MPI_Recv(rbuf, 1, sddt, 0, t*4 + c, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Send( sbuf, 1, sddt, 0, t*4+c+1, MPI_COMM_WORLD );
		}
	}

	return 0;
}

static int send_rank( int cycles, MPI_Datatype sddt, void *sbuf, char *scomp, 
		      void* rbuf, char *rcomp, int length )
{
	int outsize, do_size, done;
	double timers[trials], comp_time[trials], decomp_time[trials];

	MPI_Type_size(sddt, &outsize);
       
	size_t compressed_size, decompressed_size;
	int result;

	MPI_Barrier(MPI_COMM_WORLD);

	int c = 0;
	for( int t = 0; t < trials; t++ ){
		size_t scomp_len = (size_t)length,
               dcomp_len = (size_t)length * 2;
		
        timers[t] = MPI_Wtime();

		result = compress_buffer((unsigned char*)sbuf, outsize, (unsigned char**)&scomp, &scomp_len);

        MPI_Send( &scomp_len, 1, MPI_LONG, 1, t*4+c, MPI_COMM_WORLD );
		MPI_Send( scomp, scomp_len, MPI_BYTE, 1, t*4+c+1, MPI_COMM_WORLD);

        scomp_len = (size_t)length;
        dcomp_len = (size_t)length * 2;

        MPI_Recv( &compressed_size, 1, MPI_LONG, 1, t*4+c+2, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
        MPI_Recv( rbuf, compressed_size, MPI_BYTE, 1, t*4+c+3, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
        result = decompress_buffer(rbuf, compressed_size, (unsigned char**)&rcomp, &dcomp_len);

		timers[t] = (MPI_Wtime() - timers[t]);
    }

	print_result(outsize, trials, timers);

	return 0;
}

static int recv_rank( int cycles, MPI_Datatype sddt, void *sbuf, char *scomp, 
		      void* rbuf, char *rcomp, int length )
{
	int do_size;
	size_t compressed_size, decompressed_size;
	int result;

	MPI_Type_size( sddt, &do_size );

	MPI_Barrier(MPI_COMM_WORLD);

	int done = 1;
	int c = 0;
	for( int t = 0; t < trials; t++ ){
		size_t scomp_len = (size_t)length,
               dcomp_len = (size_t)length * 2;

		MPI_Recv( &compressed_size, 1, MPI_LONG, 0, t*4+c, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
		MPI_Recv(rbuf, compressed_size, MPI_BYTE, 0, t*4+c+1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		result = decompress_buffer(rbuf, compressed_size, (unsigned char**)&rcomp, &dcomp_len);

        scomp_len = (size_t)length;
        dcomp_len = (size_t)length * 2;

        result = compress_buffer((unsigned char*)sbuf, do_size, (unsigned char**)&scomp, &scomp_len);
        MPI_Send( &scomp_len, 1, MPI_LONG, 0, t*4+c+2, MPI_COMM_WORLD );
        MPI_Send( scomp, scomp_len, MPI_BYTE, 0, t*4+c+3, MPI_COMM_WORLD );
	}

    return 0;
}

static int send_rank_timing( int cycles, MPI_Datatype sddt, void *sbuf, char *scomp, 
		      void* rbuf, char *rcomp, int length )
{
	int outsize, do_size, done;
	double timers[trials], comp_time[trials], decomp_time[trials], timer_keep;

	MPI_Type_size(sddt, &outsize);
       
	size_t compressed_size, decompressed_size;
	int result;

	MPI_Barrier(MPI_COMM_WORLD);

	int c = 0;
	for( int t = 0; t < trials; t++ ){
		size_t scomp_len = (size_t)length,
               dcomp_len = (size_t)length * 2;
		
        comp_time[t] = MPI_Wtime();
		result = compress_buffer((unsigned char*)sbuf, outsize, (unsigned char**)&scomp, &scomp_len);
        comp_time[t] = (MPI_Wtime() - comp_time[t]);

        compressed_size = scomp_len;

        MPI_Send( &scomp_len, 1, MPI_LONG, 1, t*4+c, MPI_COMM_WORLD );

        timers[t] = MPI_Wtime();
		MPI_Send( scomp, scomp_len, MPI_BYTE, 1, t*4+c+1, MPI_COMM_WORLD);
        timers[t] = (MPI_Wtime() - timers[t]);
        timers[t] *= 2;

        scomp_len = (size_t)length;
        dcomp_len = (size_t)length * 2;

        MPI_Recv( &compressed_size, 1, MPI_LONG, 1, t*4+c+2, MPI_COMM_WORLD, MPI_STATUS_IGNORE );

//        timer_keep = MPI_Wtime();
        MPI_Recv( rbuf, compressed_size, MPI_BYTE, 1, t*4+c+3, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
//        timers[t] += MPI_Wtime() - timer_keep;
        
        decomp_time[t] = MPI_Wtime();
        result = decompress_buffer(rbuf, compressed_size, (unsigned char**)&rcomp, &dcomp_len);
        decomp_time[t] = (MPI_Wtime() - decomp_time[t]);
    }

	MPI_Type_size(sddt, &outsize);
	print_result(compressed_size, trials, comp_time);
	print_result(compressed_size, trials, decomp_time);
	print_result(compressed_size, trials, timers);
    printf("\n");

	return 0;
}

static int recv_rank_timing( int cycles, MPI_Datatype sddt, void *sbuf, char *scomp, 
		      void* rbuf, char *rcomp, int length )
{
	int do_size;
	size_t compressed_size, decompressed_size;
	int result;

	MPI_Type_size( sddt, &do_size );

	MPI_Barrier(MPI_COMM_WORLD);

	int done = 1;
	int c = 0;
	for( int t = 0; t < trials; t++ ){
		size_t scomp_len = (size_t)length,
               dcomp_len = (size_t)length * 2;

		MPI_Recv( &compressed_size, 1, MPI_LONG, 0, t*4+c, MPI_COMM_WORLD, MPI_STATUS_IGNORE );

		MPI_Datatype recv_ddt, send_ddt;

		MPI_Recv(rbuf, compressed_size, MPI_BYTE, 0, t*4+c+1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		result = decompress_buffer(rbuf, compressed_size, (unsigned char**)&rcomp, &dcomp_len);

        scomp_len = (size_t)length;
        dcomp_len = (size_t)length * 2;

        result = compress_buffer((unsigned char*)sbuf, do_size, (unsigned char**)&scomp, &scomp_len);
        MPI_Send( &scomp_len, 1, MPI_LONG, 0, t*4+c+2, MPI_COMM_WORLD );

        MPI_Send( scomp, scomp_len, MPI_BYTE, 0, t*4+c+3, MPI_COMM_WORLD );
	}

    return 0;
}

static int send_rank_compress_done( int cycles, MPI_Datatype sddt, void *sbuf, char *scomp, 
		      void* rbuf, char *rcomp, int length )
{
	int outsize, do_size, done;
	double timers[trials], comp_time[trials], decomp_time[trials];

	MPI_Type_size(sddt, &outsize);
       
	size_t compressed_size, decompressed_size;
	int result;

    size_t scomp_len = (size_t)length;
    result = compress_buffer((unsigned char*)sbuf, outsize, (unsigned char**)&scomp, &scomp_len);
	
    MPI_Barrier(MPI_COMM_WORLD);

	int c = 0;
	for( int t = 0; t < trials; t++ ){
        size_t dcomp_len = (size_t)length * 2;

        timers[t] = MPI_Wtime();

        MPI_Send( &scomp_len, 1, MPI_LONG, 1, t*4+c, MPI_COMM_WORLD );
		MPI_Send( scomp, scomp_len, MPI_BYTE, 1, t*4+c+1, MPI_COMM_WORLD);

        dcomp_len = (size_t)length * 2;

        MPI_Recv( &compressed_size, 1, MPI_LONG, 1, t*4+c+2, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
        MPI_Recv( rbuf, compressed_size, MPI_BYTE, 1, t*4+c+3, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
        result = decompress_buffer(rbuf, compressed_size, (unsigned char**)&rcomp, &dcomp_len);

		timers[t] = (MPI_Wtime() - timers[t]);
    }

	MPI_Type_size(sddt, &outsize);
	print_result(outsize, trials, timers);

	return 0;
}


static int recv_rank_compress_done( int cycles, MPI_Datatype sddt, void *sbuf, char *scomp, 
		      void* rbuf, char *rcomp, int length )
{
	int do_size;
	size_t compressed_size, decompressed_size;
	int result;
    
	MPI_Type_size( sddt, &do_size );
    size_t scomp_len = (size_t)length;
    result = compress_buffer((unsigned char*)sbuf, do_size, (unsigned char**)&scomp, &scomp_len);
    
	MPI_Barrier(MPI_COMM_WORLD);

	int done = 1;
	int c = 0;
	for( int t = 0; t < trials; t++ ){
        size_t dcomp_len = (size_t)length * 2;

        MPI_Recv( &compressed_size, 1, MPI_LONG, 0, t*4+c, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
		MPI_Recv(rbuf, compressed_size, MPI_BYTE, 0, t*4+c+1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		result = decompress_buffer(rbuf, compressed_size, (unsigned char**)&rcomp, &dcomp_len);

        MPI_Send( &scomp_len, 1, MPI_LONG, 0, t*4+c+2, MPI_COMM_WORLD );
        MPI_Send( scomp, scomp_len, MPI_BYTE, 0, t*4+c+3, MPI_COMM_WORLD );
	}

    return 0;
}

static int pingpong(int cycles, MPI_Datatype sddt, void *sbuf, char *scomp,
		    void *rbuf, char *rcomp, int length, int rank)
{
	if ( rank == 0 ){
		send_rank( cycles, sddt, sbuf, scomp, rbuf, rcomp, length );
	} else if ( rank == 1 ){
		recv_rank( cycles, sddt, sbuf, scomp, rbuf, rcomp, length );
	}
	return 0;
}

static int pingpong_compress_done(int cycles, MPI_Datatype sddt, void *sbuf, char *scomp,
		    void *rbuf, char *rcomp, int length, int rank)
{
	if ( rank == 0 ){
		send_rank_compress_done( cycles, sddt, sbuf, scomp, rbuf, rcomp, length );
	} else if ( rank == 1 ){
		recv_rank_compress_done( cycles, sddt, sbuf, scomp, rbuf, rcomp, length );
	}
	return 0;
}

static int pingpong_orig(int cycles, MPI_Datatype sddt, void *sbuf, void *rbuf, int rank)
{
	if ( rank == 0 ){
		send_rank_orig( cycles, sddt, sbuf, rbuf );
	} else if ( rank == 1 ){
		recv_rank_orig( cycles, sddt, sbuf, rbuf );
	}
	return 0;
}

static int pingpong_timing(int cycles, MPI_Datatype sddt, void *sbuf, char *scomp,
		    void *rbuf, char *rcomp, int length, int rank)
{
	if ( rank == 0 ){
		send_rank_timing( cycles, sddt, sbuf, scomp, rbuf, rcomp, length );
	} else if ( rank == 1 ){
		recv_rank_timing( cycles, sddt, sbuf, scomp, rbuf, rcomp, length );
	}
	return 0;
}

static void do_compression_test_only( int cycles, MPI_Datatype sddt, void *sbuf, char *scomp, 
       void *rbuf, char *rcomp, int length )
{

    double comp_time[trials], decomp_time[trials];

    int ddt_size;
    MPI_Type_size( sddt, &ddt_size );

    FILE *fptr = fopen("data_file", "wb");
    fwrite( sbuf, ddt_size, 1, fptr );
    fseek( fptr, 0L, SEEK_END );
    size_t sz = ftell( fptr );
    fclose(fptr);

    memset( sbuf, 0, (size_t)ddt_size );
    int *ptr = mmap( NULL, sz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, 0, 0 );
    if(ptr == MAP_FAILED){
        printf("Mapping Failed\n");
        return 1;
    }

    fptr = fopen("data_file", "r");
    if( fptr == NULL ){
        printf("Error ! opening file\n");
        exit(0);
    }

    fread( sbuf, ddt_size, 1, fptr );
    fclose( fptr );

    size_t scomp_len = (size_t)ddt_size;
    for( int t = 0; t < trials; t++ ){
        comp_time[t] = MPI_Wtime();
        scomp_len = (size_t)ddt_size;
        for( int c = 0; c < cycles; c++ ){
            compress_buffer((unsigned char*)sbuf, ddt_size, (unsigned char**)&scomp, &scomp_len);
        }
        comp_time[t] = (MPI_Wtime() - comp_time[t]) / cycles;
    }

    size_t compressed_size = scomp_len, decomp_len;
    for( int t = 0; t < trials; t++ ){
        decomp_time[t] = MPI_Wtime();
        scomp_len = (size_t)ddt_size;
        for( int c = 0; c < cycles; c++ ){
            decomp_len = length * 2;
            decompress_buffer(scomp, compressed_size, (unsigned char**)&rcomp, &decomp_len);
        }
        decomp_time[t] = (MPI_Wtime() - decomp_time[t]) / cycles;
    }

    print_result(ddt_size, trials, comp_time);
    print_result(ddt_size, trials, decomp_time);
}

static int do_test_for_ddt(int doop, MPI_Datatype sddt, MPI_Datatype rddt, int length)
{
    MPI_Aint lb, extent;
    int *sbuf, *rbuf;
    char *scomp, *rcomp;
    int i, ddt_size;

    MPI_Type_get_extent(sddt, &lb, &extent);
    MPI_Type_size( sddt, &ddt_size );

    length = ddt_size;

    sbuf = (char *) malloc(length);
    rbuf = (char *) malloc(length);

    scomp = (char *)malloc( length*2 );
    rcomp = (char *)malloc( length*2 );
    for( int i = 0; i < ddt_size/sizeof(int); i++ ){
        if( i % 4 == 0 ){
            sbuf[i] = rand();
            rbuf[i] = sbuf[i];
        }
    }

    size_t compressed_size, decompressed_size;
    int result;
    unsigned char *compressed_buffer, *decompressed_buffer;

    //result = compress_buffer(sbuf, ddt_size, &compressed_buffer, &compressed_size);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if( size == 2 ){
        MPI_Barrier(MPI_COMM_WORLD);
        pingpong_orig(cycles, sddt, (char*)sbuf, (char*)rbuf, rank);
        MPI_Barrier(MPI_COMM_WORLD);
        pingpong(cycles, sddt, (char*)sbuf, scomp, (char*)rbuf, rcomp, length, rank);
        MPI_Barrier(MPI_COMM_WORLD);
        pingpong_compress_done(cycles, sddt, (char*)sbuf, scomp, (char*)rbuf, rcomp, length, rank);
        MPI_Barrier(MPI_COMM_WORLD);
        pingpong_timing(cycles, sddt, (char*)sbuf, scomp, (char*)rbuf, rcomp, length, rank);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    if( size == 1 ){
        do_compression_test_only( cycles, sddt, sbuf, scomp, rbuf, rcomp, length );
    }

    free( scomp );
    free( rcomp );
    free(sbuf);
    free(rbuf);
    return 0;
}

int main(int argc, char *argv[])
{
    int run_tests = 0xffff; /* do all datatype tests by default */
    int rank, size;
    MPI_Datatype ddt;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if( rank == 0 && size == 2 ){
	    printf("\n! MPI_Type_contiguous(x, DOUBLE)\n");
	    printf("# Ping-pong \n");
    }
    for( int i = 4096; i < 25600000; i+=4096 ){
	    MPI_Type_contiguous( i, MPI_BYTE, &ddt );
	    MPI_Type_commit( &ddt );
	    do_test_for_ddt(run_tests, ddt, ddt, MAX_LENGTH);
	    MPI_Type_free( &ddt );
    }

    MPI_Finalize();
    exit(0);
}
