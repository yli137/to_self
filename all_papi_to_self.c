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

#include <papi.h>

#if 0 && OPEN_MPI
extern void ompi_datatype_dump( MPI_Datatype ddt );
#    define MPI_DDT_DUMP(ddt) ompi_datatype_dump((ddt))
#else
#    define MPI_DDT_DUMP(ddt)
#endif /* OPEN_MPI */

static MPI_Datatype create_ddt( int num )
{
    MPI_Datatype ddt;
    int blen[num];
    int disp[num];
    int left = 8;

    for( int i = 0; i < num; i++ ){
        disp[i] = 8 * i;
    }

    for( int i = num - 1; i >= 0; i-- ){
        if( i != 0 ){
            blen[i] = 1;
            left--;
        } else {
            blen[i] = left;
        }
    }

    MPI_Type_indexed( num, blen, disp, MPI_DOUBLE, &ddt );
    MPI_Type_create_resized( ddt, 0, 64 * 8, &ddt );
    MPI_Type_commit( &ddt );

    return ddt;
}

static MPI_Datatype create_len_ddt( int num, int len )
{
    MPI_Datatype ddt;
    int blen[num];
    int disp[num];
    int left = len * 8;

    for( int i = 0; i < num; i++ ){
        disp[i] = len * 8 * i;
    }

    for( int i = num - 1; i >= 0; i-- ){
        if( i != 0 ){
            blen[i] = len;
            left-=len;
        } else {
            blen[i] = left;
        }
    }

    MPI_Type_indexed( num, blen, disp, MPI_DOUBLE, &ddt );
    MPI_Type_commit( &ddt );

    return ddt;
}

static MPI_Datatype create_merged_contig_with_gaps(int count) /* count of the basic datatype */
{
    int array_of_blocklengths[] = {1, 1, 1};
    MPI_Aint array_of_displacements[] = {0, 8, 16};
    MPI_Datatype array_of_types[] = {MPI_DOUBLE, MPI_LONG, MPI_CHAR};
    MPI_Datatype type;

    MPI_Type_create_struct(3, array_of_blocklengths, array_of_displacements, array_of_types, &type);
    if (1 < count) {
        MPI_Datatype temp = type;
        MPI_Type_contiguous(count, temp, &type);
    }
    MPI_Type_commit(&type);
    MPI_DDT_DUMP(type);
    return type;
}

/* Create a non-contiguous resized datatype */
struct structure {
    double not_transfered;
    double transfered_1;
    double transfered_2;
};

static MPI_Datatype create_struct_constant_gap_resized_ddt(
        int number,      /* IGNORED: number of repetitions */
        int contig_size, /* IGNORED: number of elements in a contiguous chunk */
        int gap_size)    /* IGNORED: number of elements in a gap */
{
    struct structure data[1];
    MPI_Datatype struct_type, temp_type;
    MPI_Datatype types[2] = {MPI_DOUBLE, MPI_DOUBLE};
    int blocklens[2] = {1, 1};
    MPI_Aint disps[3];

    MPI_Get_address(&data[0].transfered_1, &disps[0]);
    MPI_Get_address(&data[0].transfered_2, &disps[1]);
    MPI_Get_address(&data[0], &disps[2]);
    disps[1] -= disps[2]; /*  8 */
    disps[0] -= disps[2]; /* 16 */

    MPI_Type_create_struct(2, blocklens, disps, types, &temp_type);
    MPI_Type_create_resized(temp_type, 0, sizeof(data[0]), &struct_type);
    MPI_Type_commit(&struct_type);
    MPI_Type_free(&temp_type);
    MPI_DDT_DUMP(struct_type);

    return struct_type;
}

/* Create a datatype similar to the one use by HPL */
static MPI_Datatype
create_indexed_constant_gap_ddt(int number,      /* number of repetitions */
                                int contig_size, /* number of elements in a contiguous chunk */
                                int gap_size)    /* number of elements in a gap */
{
    MPI_Datatype dt, *types;
    int i, *bLength;
    MPI_Aint *displ;

    types = (MPI_Datatype *) malloc(sizeof(MPI_Datatype) * number);
    bLength = (int *) malloc(sizeof(int) * number);
    displ = (MPI_Aint *) malloc(sizeof(MPI_Aint) * number);

    types[0] = MPI_DOUBLE;
    bLength[0] = contig_size;
    displ[0] = 0;
    for (i = 1; i < number; i++) {
        types[i] = MPI_DOUBLE;
        bLength[i] = contig_size;
        displ[i] = displ[i - 1] + sizeof(double) * (contig_size + gap_size);
    }
    MPI_Type_create_struct(number, bLength, displ, types, &dt);
    MPI_DDT_DUMP(dt);
    free(types);
    free(bLength);
    free(displ);
    MPI_Type_commit(&dt);
    return dt;
}

static MPI_Datatype create_optimized_indexed_constant_gap_ddt(
    int number,      /* number of repetitions */
    int contig_size, /* number of elements in a contiguous chunk */
    int gap_size)    /* number of elements in a gap */
{
    MPI_Datatype dt;

    MPI_Type_vector(number, contig_size, (contig_size + gap_size), MPI_DOUBLE, &dt);
    MPI_Type_commit(&dt);
    MPI_DDT_DUMP(dt);
    return dt;
}

typedef struct {
    int i[2];
    float f;
} internal_struct;
typedef struct {
    int v1;
    int gap1;
    internal_struct is[3];
} ddt_gap;

static MPI_Datatype create_indexed_gap_ddt(void)
{
    ddt_gap dt[2];
    MPI_Datatype dt1, dt2, dt3;
    int bLength[2] = {2, 1};
    MPI_Datatype types[2] = {MPI_INT, MPI_FLOAT};
    MPI_Aint displ[2];

    MPI_Get_address(&(dt[0].is[0].i[0]), &(displ[0]));
    MPI_Get_address(&(dt[0].is[0].f), &(displ[1]));
    displ[1] -= displ[0];
    displ[0] -= displ[0];
    MPI_Type_create_struct(2, bLength, displ, types, &dt1);
    /*MPI_DDT_DUMP( dt1 );*/
    MPI_Type_contiguous(3, dt1, &dt2);
    /*MPI_DDT_DUMP( dt2 );*/
    bLength[0] = 1;
    bLength[1] = 1;
    MPI_Get_address(&(dt[0].v1), &(displ[0]));
    MPI_Get_address(&(dt[0].is[0]), &(displ[1]));
    displ[1] -= displ[0];
    displ[0] -= displ[0];
    types[0] = MPI_INT;
    types[1] = dt2;
    MPI_Type_create_struct(2, bLength, displ, types, &dt3);
    /*MPI_DDT_DUMP( dt3 );*/
    MPI_Type_free(&dt1);
    MPI_Type_free(&dt2);
    MPI_Type_contiguous(10, dt3, &dt1);
    MPI_DDT_DUMP(dt1);
    MPI_Type_free(&dt3);
    MPI_Type_commit(&dt1);
    return dt1;
}

static MPI_Datatype create_indexed_gap_optimized_ddt(void)
{
    MPI_Datatype dt1, dt2, dt3;
    int bLength[3];
    MPI_Datatype types[3];
    MPI_Aint displ[3];

    MPI_Type_contiguous(40, MPI_BYTE, &dt1);
    MPI_Type_create_resized(dt1, 0, 44, &dt2);

    bLength[0] = 4;
    bLength[1] = 9;
    bLength[2] = 36;

    types[0] = MPI_BYTE;
    types[1] = dt2;
    types[2] = MPI_BYTE;

    displ[0] = 0;
    displ[1] = 8;
    displ[2] = 44 * 9 + 8;

    MPI_Type_create_struct(3, bLength, displ, types, &dt3);

    MPI_Type_free(&dt1);
    MPI_Type_free(&dt2);
    MPI_DDT_DUMP(dt3);
    MPI_Type_commit(&dt3);
    return dt3;
}

/********************************************************************
 *******************************************************************/

#define DO_CONTIG                         0x00000001
#define DO_CONSTANT_GAP                   0x00000002
#define DO_INDEXED_GAP                    0x00000004
#define DO_OPTIMIZED_INDEXED_GAP          0x00000008
#define DO_STRUCT_CONSTANT_GAP_RESIZED    0x00000010
#define DO_STRUCT_MERGED_WITH_GAP_RESIZED 0x00000020

#define DO_PACK        0x01000000
#define DO_UNPACK      0x02000000
#define DO_ISEND_RECV  0x04000000
#define DO_ISEND_IRECV 0x08000000
#define DO_IRECV_SEND  0x10000000
#define DO_IRECV_ISEND 0x20000000

#define MIN_LENGTH 1024
#define MAX_LENGTH (1024 * 1024)

static int cycles = 10;
static int trials = 10;
static int warmups = 0;

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

    int NUM_EVENTS = 55;
    int all_events[55] = {PAPI_L1_DCM, PAPI_L1_ICM, PAPI_L2_DCM, PAPI_L2_ICM, PAPI_L1_TCM,
                        PAPI_L2_TCM, PAPI_L3_TCM, PAPI_CA_SNP, PAPI_CA_SHR, PAPI_CA_CLN,
                        PAPI_CA_ITV, PAPI_L3_LDM, PAPI_TLB_DM, PAPI_TLB_IM,
                        PAPI_L1_LDM, PAPI_L1_STM, PAPI_L2_LDM, PAPI_L2_STM, PAPI_PRF_DM,
                        PAPI_MEM_WCY, PAPI_STL_ICY, PAPI_FUL_ICY, PAPI_STL_CCY, PAPI_FUL_CCY,
                        PAPI_BR_UCN, PAPI_BR_CN, PAPI_BR_TKN, PAPI_BR_NTK, PAPI_BR_MSP,
                        PAPI_BR_PRC, PAPI_TOT_INS, PAPI_LD_INS, PAPI_SR_INS, PAPI_BR_INS,
                        PAPI_RES_STL, PAPI_TOT_CYC, PAPI_LST_INS, PAPI_L2_DCA, PAPI_L3_DCA,
                        PAPI_L2_DCR, PAPI_L3_DCR, PAPI_L2_DCW, PAPI_L3_DCW, PAPI_L2_ICH,
                        PAPI_L2_ICA, PAPI_L3_ICA, PAPI_L2_ICR, PAPI_L3_ICR, PAPI_L2_TCA,
                        PAPI_L3_TCA, PAPI_L2_TCR, PAPI_L3_TCR, PAPI_L2_TCW, PAPI_L3_TCW,
                        PAPI_REF_CYC };

    int retval;
    retval = PAPI_library_init(PAPI_VER_CURRENT);
    long long event_values[NUM_EVENTS];
    {
	    for( int i = 0; i < NUM_EVENTS; i++ )
		    event_values[i] = 0;
    }

    int j = 0;
    for( ; j < NUM_EVENTS; j++ ){
	    for( int q = 0; q < trials; q++ )
		    timers[q] = 0;
	    
	    int num_events = 1;
	    int events[num_events];
	    
	    events[0] = all_events[j];

	    int eventset = PAPI_NULL;
	    long long values[num_events];

	    retval = PAPI_create_eventset(&eventset);
	    if( retval != 0 ){
		    printf("Error on %d event\n", j);
		    exit(0);
	    }
	    retval = PAPI_add_events(eventset, events, num_events);
	    if( retval != 0 ){
		    printf("Error on %d event\n", j);
		    exit(0);
	    }


	    for (t = 0; t < warmups; t++) {
		    for (c = 0; c < cycles; c++) {
			    position = 0;
			    MPI_Pack(sbuf, scount, sdt, packed_buf, outsize, &position, MPI_COMM_WORLD);
		    }
	    }

	    retval = PAPI_start( eventset );
	    if( retval != 0 ){
		    printf("Error on %d event\n", j);
		    exit(0);
	    }
	    for (t = 0; t < trials; t++) {
		    timers[t] = MPI_Wtime();
		    for (c = 0; c < cycles; c++) {
			    position = 0;
			    MPI_Pack(sbuf, scount, sdt, packed_buf, outsize, &position, MPI_COMM_WORLD);
		    }
		    timers[t] = (MPI_Wtime() - timers[t]) / cycles;
	    }
	    retval = PAPI_stop( eventset, values );
	    if( retval != 0 ){
		    printf("Error on %d event\n", j);
		    exit(0);
	    }

	    event_values[j] = values[0];
	    PAPI_cleanup_eventset( &eventset );
	    PAPI_destroy_eventset( &eventset );
    }

    print_result(outsize, trials, timers);
    for( int j = 0; j < NUM_EVENTS; j++ )
	    printf("%lld ", event_values[j] / trials / cycles );
    printf("\n");

    PAPI_shutdown();


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

static int isend_recv(int cycles, MPI_Datatype sdt, int scount, void *sbuf, MPI_Datatype rdt,
                      int rcount, void *rbuf)
{
    int myself, tag = 0, c, t, slength, rlength;
    MPI_Status status;
    MPI_Request req;
    double timers[trials];

    MPI_Type_size(sdt, &slength);
    slength *= scount;
    MPI_Type_size(rdt, &rlength);
    rlength *= rcount;

    MPI_Comm_rank(MPI_COMM_WORLD, &myself);

    for (t = 0; t < trials; t++) {
        timers[t] = MPI_Wtime();
        for (c = 0; c < cycles; c++) {
            MPI_Isend(sbuf, scount, sdt, myself, tag, MPI_COMM_WORLD, &req);
            MPI_Recv(rbuf, rcount, rdt, myself, tag, MPI_COMM_WORLD, &status);
            MPI_Wait(&req, &status);
        }
        timers[t] = (MPI_Wtime() - timers[t]) / cycles;
    }
    print_result(rlength, trials, timers);
    return 0;
}

static int irecv_send(int cycles, MPI_Datatype sdt, int scount, void *sbuf, MPI_Datatype rdt,
                      int rcount, void *rbuf)
{
    int myself, tag = 0, c, t, slength, rlength;
    MPI_Request req;
    MPI_Status status;
    double timers[trials];

    MPI_Type_size(sdt, &slength);
    slength *= scount;
    MPI_Type_size(rdt, &rlength);
    rlength *= rcount;

    MPI_Comm_rank(MPI_COMM_WORLD, &myself);

    for (t = 0; t < trials; t++) {
        timers[t] = MPI_Wtime();
        for (c = 0; c < cycles; c++) {
            MPI_Irecv(rbuf, rcount, rdt, myself, tag, MPI_COMM_WORLD, &req);
            MPI_Send(sbuf, scount, sdt, myself, tag, MPI_COMM_WORLD);
            MPI_Wait(&req, &status);
        }
        timers[t] = (MPI_Wtime() - timers[t]) / cycles;
    }
    print_result(rlength, trials, timers);
    return 0;
}

static int isend_irecv_wait(int cycles, MPI_Datatype sdt, int scount, void *sbuf, MPI_Datatype rdt,
                            int rcount, void *rbuf)
{
    int myself, tag = 0, c, t, slength, rlength;
    MPI_Request requests[2];
    MPI_Status statuses[2];
    double timers[trials];

    MPI_Type_size(sdt, &slength);
    slength *= scount;
    MPI_Type_size(rdt, &rlength);
    rlength *= rcount;

    MPI_Comm_rank(MPI_COMM_WORLD, &myself);

    for (t = 0; t < trials; t++) {
        timers[t] = MPI_Wtime();
        for (c = 0; c < cycles; c++) {
            MPI_Isend(sbuf, scount, sdt, myself, tag, MPI_COMM_WORLD, &requests[0]);
            MPI_Irecv(rbuf, rcount, rdt, myself, tag, MPI_COMM_WORLD, &requests[1]);
            MPI_Waitall(2, requests, statuses);
        }
        timers[t] = (MPI_Wtime() - timers[t]) / cycles;
    }
    print_result(rlength, trials, timers);
    return 0;
}

static int irecv_isend_wait(int cycles, MPI_Datatype sdt, int scount, void *sbuf, MPI_Datatype rdt,
                            int rcount, void *rbuf)
{
    int myself, tag = 0, c, t, slength, rlength;
    MPI_Request requests[2];
    MPI_Status statuses[2];
    double timers[trials];

    MPI_Type_size(sdt, &slength);
    slength *= scount;
    MPI_Type_size(rdt, &rlength);
    rlength *= rcount;

    MPI_Comm_rank(MPI_COMM_WORLD, &myself);

    for (t = 0; t < trials; t++) {
        timers[t] = MPI_Wtime();
        for (c = 0; c < cycles; c++) {
            MPI_Irecv(rbuf, rcount, rdt, myself, tag, MPI_COMM_WORLD, &requests[0]);
            MPI_Isend(sbuf, scount, sdt, myself, tag, MPI_COMM_WORLD, &requests[1]);
            MPI_Waitall(2, requests, statuses);
        }
        timers[t] = (MPI_Wtime() - timers[t]) / cycles;
    }
    print_result(rlength, trials, timers);
    return 0;
}

static int do_test_for_ddt(int doop, MPI_Datatype sddt, MPI_Datatype rddt, int length)
{
    MPI_Aint lb, extent;
    char *sbuf, *rbuf;
    int i, ddt_size;

    MPI_Type_get_extent(sddt, &lb, &extent);
    MPI_Type_size( sddt, &ddt_size );

    length = 25600000 / ddt_size * extent;

    sbuf = (char *) malloc(length);
    rbuf = (char *) malloc(length);
    if (doop & DO_PACK) {
        printf("# Pack (max length %d)\n", length);
        for (i = 1; i <= (length / extent); i *= 2) {
            pack(cycles, sddt, i, sbuf, rbuf);
        }
    }

    if (doop & DO_UNPACK) {
        printf("# Unpack (length %d)\n", length);
        for (i = 1; i <= (length / extent); i *= 2) {
            unpack(cycles, sbuf, rddt, i, rbuf);
        }
    }

    if (doop & DO_ISEND_RECV) {
        printf("# Isend recv (length %d)\n", length);
        for (i = 1; i <= (length / extent); i *= 2) {
            isend_recv(cycles, sddt, i, sbuf, rddt, i, rbuf);
        }
    }

    if (doop & DO_ISEND_IRECV) {
        printf("# Isend Irecv Wait (length %d)\n", length);
        for (i = 1; i <= (length / extent); i *= 2) {
            isend_irecv_wait(cycles, sddt, i, sbuf, rddt, i, rbuf);
        }
    }

    if (doop & DO_IRECV_SEND) {
        printf("# Irecv send (length %d)\n", length);
        for (i = 1; i <= (length / extent); i *= 2) {
            irecv_send(cycles, sddt, i, sbuf, rddt, i, rbuf);
        }
    }

    if (doop & DO_IRECV_SEND) {
        printf("# Irecv Isend Wait (length %d)\n", length);
        for (i = 1; i <= (length / extent); i *= 2) {
            irecv_isend_wait(cycles, sddt, i, sbuf, rddt, i, rbuf);
        }
    }
    free(sbuf);
    free(rbuf);
    return 0;
}

static int do_pipeline_pack( const void *inbuf, int incount, MPI_Datatype datatype,
        void *outbuf, int outsize, int pipe_size )
{
    int position, myself, c, t, i, ddt_size;
    double timers[trials];
    size_t extent, lb;

    int pc, check;
    int j = 0, current_pos = 0, keep_pos = 0,
        keep, do_count;

    MPI_Type_get_extent( datatype, &lb, &extent );
    MPI_Type_size( datatype, &ddt_size );

    MPI_Comm_rank( MPI_COMM_WORLD, &myself );

    for( t = 0; t < trials; t++ ) {
        timers[t] = MPI_Wtime();

        for( c = 0; c < cycles; c++ ) {
            do_count = incount;
            keep = 0;
            for( int k = 0 ; k < incount / pipe_size; k++ ){
                current_pos = 0;
                MPI_Pack( inbuf + k * pipe_size * extent,
                        pipe_size,
                        datatype,
                        outbuf + k * pipe_size * ddt_size,
                        pipe_size * ddt_size,
                        &current_pos,
                        MPI_COMM_SELF );
                keep += pipe_size;
                do_count -= pipe_size;
            }

            if( do_count != 0 ){
                current_pos = 0;
                MPI_Pack( inbuf + keep * pipe_size * extent,
                        do_count,
                        datatype,
                        outbuf + keep * pipe_size * ddt_size,
                        do_count * ddt_size,
                        &current_pos,
                        MPI_COMM_SELF );
            }

        }
        timers[t] = (MPI_Wtime() - timers[t]) / cycles;
    }

    print_result(outsize, trials, timers);

    return 0;
}

static int pack_pipeline( int cycles,
        MPI_Datatype sdt, int scount, void* sbuf,
        void* packed_buf, int pipe_size )
{
    int position, myself, c, t, outsize;
    double timers[trials];

    MPI_Type_size( sdt, &outsize );
    outsize *= scount;

    do_pipeline_pack(sbuf, scount, sdt, packed_buf, outsize, pipe_size );

    return 0;
}

static int do_pipeline_test_for_ddt( int doop, MPI_Datatype sddt, MPI_Datatype rddt, size_t length, int num )
{
    MPI_Aint lb, extent;
    char *sbuf, *rbuf;
    int i;
    int max_length;

    MPI_Type_get_extent( sddt, &lb, &extent );
    MPI_Type_size( sddt, &max_length );

    length = 16800000. / max_length * extent;

    sbuf = (char*)malloc( length );
    rbuf = (char*)malloc( length );

    printf("\n# Pack (max length %zu) Pipeline %d ddt per segment\n",
            length,
            num );

    for( i = num; i < (length / extent); i*=2 ){
        pack_pipeline( cycles, sddt, i, sbuf, rbuf, num );
    }


    free( sbuf );
    free( rbuf );
    return 0;
}

int main(int argc, char *argv[])
{
    int run_tests = 0xffff; /* do all datatype tests by default */
    int rank, size;
    MPI_Datatype ddt;

    run_tests |= DO_PACK;// | DO_UNPACK;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank != 0) {
        MPI_Finalize();
        exit(0);
    }

    if (run_tests & DO_OPTIMIZED_INDEXED_GAP) {
        for( int i = 2; i < 9; i++ ){

            i = 8;       
            if( i == 2 )
                printf("\n! 7_1 ddt\n\n");
            if( i == 3 )
                printf("\n! 6_1_1 ddt\n\n");
            if( i == 4 )
                printf("\n! 5_1_1_1 ddt\n\n");
            if( i == 5 )
                printf("\n! 4_1_1_1_1 ddt\n\n");
            if( i == 6 )
                printf("\n! 3_1_1_1_1_1 ddt\n\n");
            if( i == 7 )
                printf("\n! 2_1_1_1_1_1_1 ddt\n\n");
            if( i == 8 )
                printf("\n! 1_1_1_1_1_1_1_1 ddt\n\n");
            

            ddt = create_ddt( i );
            MPI_DDT_DUMP(ddt);
            do_test_for_ddt(run_tests, ddt, ddt, MAX_LENGTH);

            size_t true_lb, true_extent,
                   lb, extent;
            MPI_Type_get_true_extent( ddt, &true_lb, &true_extent );
            MPI_Type_get_extent( ddt, &lb, &extent );

            size_t true_distance = (true_extent - true_lb) / 64 * 64 + 3 * 64,
                   distance = (extent - lb) / 64 * 64 + 3 * 64;

            if( true_distance <= distance ){
                //printf("\n# pack calculated pipeline size (data size) %d\n", true_distance);
//                do_pipeline_test_for_ddt( run_tests, ddt, ddt, MAX_LENGTH, 32000 / true_distance );
            } else {
                //printf("\n# pack calculated pipeline size (data size) %d\n", distance);
//                do_pipeline_test_for_ddt( run_tests, ddt, ddt, MAX_LENGTH, 32000 / distance );
            }


            for( int j = 8; j < 4097; j *= 2 ){
                //printf("\n# pack pipeline size (data size) %d\n\n", j * extent);
//                do_pipeline_test_for_ddt( run_tests, ddt, ddt, MAX_LENGTH, j );
            }

            MPI_Type_free(&ddt);
        }
    }

    printf("\n! indexed gap\n\n");
    ddt = create_indexed_gap_ddt();
    MPI_DDT_DUMP(ddt);
    do_test_for_ddt(run_tests, ddt, ddt, MAX_LENGTH);
    MPI_Type_free(&ddt);

    printf("\n! optimized indexed gap\n\n");
    ddt = create_indexed_gap_optimized_ddt();
    MPI_DDT_DUMP(ddt);
    do_test_for_ddt(run_tests, ddt, ddt, MAX_LENGTH);
    MPI_Type_free(&ddt);

    printf("\n! constant indexed gap\n\n");
    ddt = create_indexed_constant_gap_ddt(80, 100, 1);
    MPI_DDT_DUMP(ddt);
    do_test_for_ddt(run_tests, ddt, ddt, MAX_LENGTH);
    MPI_Type_free(&ddt);

    printf("\n! optimized constant indexed gap\n\n");
    ddt = create_optimized_indexed_constant_gap_ddt(80, 100, 1);
    MPI_DDT_DUMP(ddt);
    do_test_for_ddt(run_tests, ddt, ddt, MAX_LENGTH);
    MPI_Type_free(&ddt);

    printf("\n! struct constant gap resized\n\n");
    ddt = create_merged_contig_with_gaps(1);
    MPI_DDT_DUMP(ddt);
    do_test_for_ddt(run_tests, ddt, ddt, MAX_LENGTH);
    MPI_Type_free(&ddt);

    MPI_Finalize();
    exit(0);
}
