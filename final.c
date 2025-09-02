#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>

#define IDX(x, y, z, t, X, Y, Z, NC) ((((z)*(Y)+(y))*(X)+(x))*(NC)+(t))

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 10) {
        if (rank == 0) printf("Usage: %s input_file PX PY PZ NX NY NZ NC output_file\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    char *input_file = argv[1];
    int PX = atoi(argv[2]), PY = atoi(argv[3]), PZ = atoi(argv[4]);
    int NX = atoi(argv[5]), NY = atoi(argv[6]), NZ = atoi(argv[7]), NC = atoi(argv[8]);
    char *output_file = argv[9];

    if (PX * PY * PZ != size) {
        if (rank == 0) printf("Error: PX * PY * PZ must equal number of processes\n");
        MPI_Finalize();
        return 1;
    }

    int local_nx = NX / PX;
    int local_ny = NY / PY;
    int local_nz = NZ / PZ;
    int halo_nx = local_nx + 2;
    int halo_ny = local_ny + 2;
    int halo_nz = local_nz + 2;

    int rx = rank % PX;
    int ry = (rank / PX) % PY;
    int rz = rank / (PX * PY);
     
    double t1 = MPI_Wtime();

    float *core = malloc(local_nx * local_ny * local_nz * NC * sizeof(float));
    float *halo = malloc(halo_nx * halo_ny * halo_nz * NC * sizeof(float));
    for (int i = 0; i < halo_nx * halo_ny * halo_nz * NC; i++) halo[i] = NAN;

    MPI_File fh;
    MPI_File_open(MPI_COMM_WORLD, input_file, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

    int gsizes[4] = {NZ, NY, NX, NC};
    int lsizes[4] = {local_nz, local_ny, local_nx, NC};
    int starts[4] = {rz * local_nz, ry * local_ny, rx * local_nx, 0};

    MPI_Datatype filetype;
    MPI_Type_create_subarray(4, gsizes, lsizes, starts, MPI_ORDER_C, MPI_FLOAT, &filetype);
    MPI_Type_commit(&filetype);
    MPI_File_set_view(fh, 0, MPI_FLOAT, filetype, "native", MPI_INFO_NULL);
    MPI_File_read_all(fh, core, local_nx * local_ny * local_nz * NC, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);
    MPI_Type_free(&filetype);
    for (int z = 0; z < local_nz; z++)
        for (int y = 0; y < local_ny; y++)
            for (int x = 0; x < local_nx; x++)
                for (int t = 0; t < NC; t++) {
                    int src = IDX(x, y, z, t, local_nx, local_ny, local_nz, NC);
                    int dst = IDX(x + 1, y + 1, z + 1, t, halo_nx, halo_ny, halo_nz, NC);
                    halo[dst] = core[src];
                }
     double t2 = MPI_Wtime();

    int x_minus = (rx > 0) ? rank - 1 : MPI_PROC_NULL;
    int x_plus  = (rx < PX - 1) ? rank + 1 : MPI_PROC_NULL;
    int y_minus = (ry > 0) ? rank - PX : MPI_PROC_NULL;
    int y_plus  = (ry < PY - 1) ? rank + PX : MPI_PROC_NULL;
    int z_minus = (rz > 0) ? rank - PX * PY : MPI_PROC_NULL;
    int z_plus  = (rz < PZ - 1) ? rank + PX * PY : MPI_PROC_NULL;

    int sizes[4] = {halo_nz, halo_ny, halo_nx, NC};
    int subsizes_x[4] = {local_nz, local_ny, 1, NC};
    int subsizes_y[4] = {local_nz, 1, local_nx, NC};
    int subsizes_z[4] = {1, local_ny, local_nx, NC};

    MPI_Datatype x_send_l, x_recv_r, x_send_r, x_recv_l;
    MPI_Datatype y_send_f, y_recv_b, y_send_b, y_recv_f;
    MPI_Datatype z_send_d, z_recv_u, z_send_u, z_recv_d;

    int start_l[4] = {1, 1, 1, 0}, recv_r[4] = {1, 1, halo_nx - 1, 0};
    int start_r[4] = {1, 1, local_nx, 0}, recv_l[4] = {1, 1, 0, 0};

    MPI_Type_create_subarray(4, sizes, subsizes_x, start_l, MPI_ORDER_C, MPI_FLOAT, &x_send_l);
    MPI_Type_create_subarray(4, sizes, subsizes_x, recv_r, MPI_ORDER_C, MPI_FLOAT, &x_recv_r);
    MPI_Type_create_subarray(4, sizes, subsizes_x, start_r, MPI_ORDER_C, MPI_FLOAT, &x_send_r);
    MPI_Type_create_subarray(4, sizes, subsizes_x, recv_l, MPI_ORDER_C, MPI_FLOAT, &x_recv_l);

    MPI_Type_commit(&x_send_l); MPI_Type_commit(&x_recv_r);
    MPI_Type_commit(&x_send_r); MPI_Type_commit(&x_recv_l);

    MPI_Sendrecv(halo, 1, x_send_l, x_minus, 0,
                 halo, 1, x_recv_r, x_plus, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(halo, 1, x_send_r, x_plus, 1,
                 halo, 1, x_recv_l, x_minus, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    int start_yf[4] = {1, 1, 1, 0}, recv_yb[4] = {1, halo_ny - 1, 1, 0};
    int start_yb[4] = {1, local_ny, 1, 0}, recv_yf[4] = {1, 0, 1, 0};

    MPI_Type_create_subarray(4, sizes, subsizes_y, start_yf, MPI_ORDER_C, MPI_FLOAT, &y_send_f);
    MPI_Type_create_subarray(4, sizes, subsizes_y, recv_yb, MPI_ORDER_C, MPI_FLOAT, &y_recv_b);
    MPI_Type_create_subarray(4, sizes, subsizes_y, start_yb, MPI_ORDER_C, MPI_FLOAT, &y_send_b);
    MPI_Type_create_subarray(4, sizes, subsizes_y, recv_yf, MPI_ORDER_C, MPI_FLOAT, &y_recv_f);

    MPI_Type_commit(&y_send_f); MPI_Type_commit(&y_recv_b);
    MPI_Type_commit(&y_send_b); MPI_Type_commit(&y_recv_f);

    MPI_Sendrecv(halo, 1, y_send_f, y_minus, 2,
                 halo, 1, y_recv_b, y_plus, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(halo, 1, y_send_b, y_plus, 3,
                 halo, 1, y_recv_f, y_minus, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    int start_zd[4] = {1, 1, 1, 0}, recv_zu[4] = {halo_nz - 1, 1, 1, 0};
    int start_zu[4] = {local_nz, 1, 1, 0}, recv_zd[4] = {0, 1, 1, 0};

    MPI_Type_create_subarray(4, sizes, subsizes_z, start_zd, MPI_ORDER_C, MPI_FLOAT, &z_send_d);
    MPI_Type_create_subarray(4, sizes, subsizes_z, recv_zu, MPI_ORDER_C, MPI_FLOAT, &z_recv_u);
    MPI_Type_create_subarray(4, sizes, subsizes_z, start_zu, MPI_ORDER_C, MPI_FLOAT, &z_send_u);
    MPI_Type_create_subarray(4, sizes, subsizes_z, recv_zd, MPI_ORDER_C, MPI_FLOAT, &z_recv_d);

    MPI_Type_commit(&z_send_d); MPI_Type_commit(&z_recv_u);
    MPI_Type_commit(&z_send_u); MPI_Type_commit(&z_recv_d);

    MPI_Sendrecv(halo, 1, z_send_d, z_minus, 4,
                 halo, 1, z_recv_u, z_plus, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(halo, 1, z_send_u, z_plus, 5,
                 halo, 1, z_recv_d, z_minus, 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    MPI_Type_free(&x_send_l); MPI_Type_free(&x_recv_r);
    MPI_Type_free(&x_send_r); MPI_Type_free(&x_recv_l);
    MPI_Type_free(&y_send_f); MPI_Type_free(&y_recv_b);
    MPI_Type_free(&y_send_b); MPI_Type_free(&y_recv_f);
    MPI_Type_free(&z_send_d); MPI_Type_free(&z_recv_u);
    MPI_Type_free(&z_send_u); MPI_Type_free(&z_recv_d);

    MPI_Barrier(MPI_COMM_WORLD);
    // Computation
    int *local_minima = calloc(NC, sizeof(int));
    int *local_maxima = calloc(NC, sizeof(int));
    float *local_gmin = malloc(NC * sizeof(float));
    float *local_gmax = malloc(NC * sizeof(float));
    for (int t = 0; t < NC; t++) {
        local_gmin[t] = FLT_MAX;
        local_gmax[t] = -FLT_MAX;
        for (int z = 1; z <= local_nz; z++) {
            for (int y = 1; y <= local_ny; y++) {
                for (int x = 1; x <= local_nx; x++) {
                    int idx = IDX(x, y, z, t, halo_nx, halo_ny, halo_nz, NC);
                    float center = halo[idx];
                    float n[6] = {
                        halo[IDX(x+1, y, z, t, halo_nx, halo_ny, halo_nz, NC)],
                        halo[IDX(x-1, y, z, t, halo_nx, halo_ny, halo_nz, NC)],
                        halo[IDX(x, y+1, z, t, halo_nx, halo_ny, halo_nz, NC)],
                        halo[IDX(x, y-1, z, t, halo_nx, halo_ny, halo_nz, NC)],
                        halo[IDX(x, y, z+1, t, halo_nx, halo_ny, halo_nz, NC)],
                        halo[IDX(x, y, z-1, t, halo_nx, halo_ny, halo_nz, NC)]
                    };
                    int is_min = 1, is_max = 1;
                    for (int i = 0; i < 6; i++) {
                        if (!isnan(n[i])) {
                            if (center >= n[i]) is_min = 0;
                            if (center <= n[i]) is_max = 0;
                        }
                    }
                    if (center <= local_gmin[t]) local_gmin[t] = center;
                    if (center >= local_gmax[t]) local_gmax[t] = center;
                    if (is_min) local_minima[t]++;
                    if (is_max) local_maxima[t]++;
                }
            }
        }
    }
    double t3 = MPI_Wtime();
    int *global_minima = NULL, *global_maxima = NULL;
    float *global_gmin = NULL, *global_gmax = NULL;
    if (rank == 0) {
        global_minima = malloc(NC * sizeof(int));
        global_maxima = malloc(NC * sizeof(int));
        global_gmin = malloc(NC * sizeof(float));
        global_gmax = malloc(NC * sizeof(float));
    }

    MPI_Reduce(local_minima, global_minima, NC, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_maxima, global_maxima, NC, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_gmin, global_gmin, NC, MPI_FLOAT, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_gmax, global_gmax, NC, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);

    double read_time = t2 - t1;
    double comp_time = t3 - t2;
    double total_time = t3 - t1;
    double max_read, max_comp, max_total;
    MPI_Reduce(&read_time, &max_read, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comp_time, &max_comp, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_time, &max_total, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        FILE *out = fopen(output_file, "w");
        for (int t = 0; t < NC; t++)
            fprintf(out, "(%d,%d)%c", global_minima[t], global_maxima[t], (t == NC - 1 ? '\n' : ' '));
        for (int t = 0; t < NC; t++)
            fprintf(out, "(%.6f,%.6f)%c", global_gmin[t], global_gmax[t], (t == NC - 1 ? '\n' : ' '));
        fprintf(out, "%.6f %.6f %.6f\n", max_read, max_comp, max_total);
        fclose(out);
    }

    free(core); free(halo);
    free(local_minima); free(local_maxima); free(local_gmin); free(local_gmax);
    if (rank == 0) {
        free(global_minima); free(global_maxima); free(global_gmin); free(global_gmax);
    }

    MPI_Finalize();
    return 0;
}

