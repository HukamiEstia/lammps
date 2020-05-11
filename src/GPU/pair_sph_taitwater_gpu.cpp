
#include "pair_sph_taitwater_gpu.h"
#include <cmath>
#include "atom.h"
#include "force.h"
#include "comm.h"
#include "neigh_list.h"
#include "memory.h"
#include "error.h"
#include "domain.h"
#include <cstdio>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "atom_vec.h"
#include "neighbor.h"
#include "integrate.h"
#include "neigh_request.h"
#include "universe.h"
#include "update.h"
#include "gpu_extra.h"
#include "suffix.h"
#include "math_const.h"
using namespace LAMMPS_NS;

int sph_taitwater_gpu_init(const int ntypes, double **cutsq, double *B,double **viscosity ,double *rho0,double **cut, double *soundspeed,double *special_lj, const int nlocal,
                   const int nall, const int max_nbors, const int maxspecial,
                   const double cell_size, int &gpu_mode, FILE *screen);

void sph_taitwater_gpu_reinit(const int ntypes, double **cutsq, double **host_viscosity,double **host_cut);
				   
void sph_taitwater_gpu_clear();
int ** sph_taitwater_gpu_compute_n(const int ago, const int inum,
                           const int nall, double **host_x, double **host_v,int *host_type, double *sublo, double *subhi, tagint *tag, int **nspecial,
                           tagint **special, const bool eflag, const bool vflag,
                           const bool eatom, const bool vatom, int &host_start,
                           int **ilist, int **jnum,
                           const double cpu_time, bool &success);
void sph_taitwater_gpu_compute(const int ago, const int inum, const int nall,
                       double **host_x,double **host_v,int *host_type, int *ilist, int *numj,
                       int **firstneigh, const bool eflag, const bool vflag,
                       const bool eatom, const bool vatom, int &host_start,
                       const double cpu_time, bool &success);
double sph_taitwater_gpu_bytes();

using namespace MathConst;

/* ---------------------------------------------------------------------- */

PairSPHTaitwaterGPU::PairSPHTaitwaterGPU(LAMMPS *lmp) : PairSPHTaitwater(lmp), gpu_mode(GPU_FORCE)
{
 respa_enable = 0;
  cpu_time = 0.0;
  suffix_flag |= Suffix::GPU;
  GPU_EXTRA::gpu_ready(lmp->modify, lmp->error);
}

/* ---------------------------------------------------------------------- */

PairSPHTaitwaterGPU::~PairSPHTaitwaterGPU() {
  sph_taitwater_gpu_clear();
}

/* ---------------------------------------------------------------------- */

void PairSPHTaitwaterGPU::compute(int eflag, int vflag) {
  ev_init(eflag,vflag);

  int nall = atom->nlocal + atom->nghost;
  int inum, host_start;

  bool success = true;
  int *ilist, *numneigh, **firstneigh;
  if (gpu_mode != GPU_FORCE) {
    inum = atom->nlocal;
    firstneigh = sph_taitwater_gpu_compute_n(neighbor->ago, inum, nall,
                                     atom->x,atom->vest, atom->type, domain->sublo,
                                     domain->subhi, atom->tag, atom->nspecial,
                                     atom->special, eflag, vflag, eflag_atom,
                                     vflag_atom, host_start,
                                     &ilist, &numneigh, cpu_time, success);
  } else {
    inum = list->inum;
    ilist = list->ilist;
    numneigh = list->numneigh;
    firstneigh = list->firstneigh;
	
    sph_taitwater_gpu_compute(neighbor->ago, inum, nall, atom->x,atom->vest, atom->type,
                      ilist, numneigh, firstneigh, eflag, vflag, eflag_atom,
                      vflag_atom, host_start, cpu_time, success);
  }
  if (!success)
    error->one(FLERR,"Insufficient memory on accelerator");

  if (host_start<inum) {
    cpu_time = MPI_Wtime();
    cpu_compute(host_start, inum, eflag, vflag, ilist, numneigh, firstneigh);
    cpu_time = MPI_Wtime() - cpu_time;
  }
  
}



/* ----------------------------------------------------------------------
 init for one type pair i,j and corresponding j,i
 ------------------------------------------------------------------------- */

void PairSPHTaitwaterGPU::init_style() {

  if (force->newton_pair)
    error->all(FLERR,"Cannot use newton pair with sph/taitwater/gpu pair style");

  // Repeat cutsq calculation because done after call to init_style
  double maxcut = -1.0;
  double mcut;
  for (int i = 1; i <= atom->ntypes; i++) {
    for (int j = i; j <= atom->ntypes; j++) {
      if (setflag[i][j] != 0 || (setflag[i][i] != 0 && setflag[j][j] != 0)) {
        mcut = init_one(i,j);
        mcut *= mcut;
        if (mcut > maxcut)
          maxcut = mcut;
        cutsq[i][j] = cutsq[j][i] = mcut;
      } else
        cutsq[i][j] = cutsq[j][i] = 0.0;
    }
  }
  double cell_size = sqrt(maxcut) + neighbor->skin;

  int maxspecial=0;
  if (atom->molecular)
    maxspecial=atom->maxspecial;
  int success = sph_taitwater_gpu_init(atom->ntypes+1, cutsq, B, viscosity,atom->rho,cut,soundspeed, force->special_lj,atom->nlocal,
                              atom->nlocal+atom->nghost, 300, maxspecial,
                              cell_size, gpu_mode, screen);
  GPU_EXTRA::check_flag(success,error,world);

  if (gpu_mode == GPU_FORCE) {
    int irequest = neighbor->request(this,instance_me);
    neighbor->requests[irequest]->half = 0;
    neighbor->requests[irequest]->full = 1;
  }
}

void PairSPHTaitwaterGPU::reinit()
{
  Pair::reinit();

  sph_taitwater_gpu_reinit(atom->ntypes+1, cutsq, viscosity,cut);
}

double PairSPHTaitwaterGPU::memory_usage()
{
  double bytes = Pair::memory_usage();
  return bytes + sph_taitwater_gpu_bytes();
}
void PairSPHTaitwaterGPU::cpu_compute(int start, int inum, int eflag, int /* vflag */,
                              int *ilist, int *numneigh, int **firstneigh) {
	 int i, j, ii, jj, jnum, itype, jtype;
  double xtmp, ytmp, ztmp, delx, dely, delz, fpair;

  int  *jlist;
  double vxtmp, vytmp, vztmp, imass, jmass, fi, fj, fvisc, h, ih, ihsq;
  double rsq, tmp, wfd, delVdotDelR, mu, deltaE;

  

  double **v = atom->vest;
  double **x = atom->x;
  double **f = atom->f;
  double *rho = atom->rho;
  double *mass = atom->mass;
  double *de = atom->de;
  double *drho = atom->drho;
     double    *cv = atom-> cv;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  
  /***************************************************/
  double *e = atom->e;
  /****************************************************/

  // check consistency of pair coefficients

  if (first) {
    for (i = 1; i <= atom->ntypes; i++) {
      for (j = 1; i <= atom->ntypes; i++) {
        if (cutsq[i][j] > 1.e-32) {
          if (!setflag[i][i] || !setflag[j][j]) {
            if (comm->me == 0) {
              printf(
                  "SPH particle types %d and %d interact with cutoff=%g, but not all of their single particle properties are set.\n",
                  i, j, sqrt(cutsq[i][j]));
            }
          }
        }
      }
    }
    first = 0;
  }

  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    vxtmp = v[i][0];
    vytmp = v[i][1];
    vztmp = v[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    imass = mass[itype];

    // compute pressure of atom i with Tait EOS
    tmp = rho[i] / rho0[itype];
    fi = tmp * tmp * tmp;
    fi = B[itype] * (fi * fi * tmp - 1.0) / (rho[i] * rho[i]);

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx * delx + dely * dely + delz * delz;
      jtype = type[j];
      jmass = mass[jtype];

      if (rsq < cutsq[itype][jtype]) {

        h = cut[itype][jtype];
        ih = 1.0 / h;
        ihsq = ih * ih;

        wfd = h - sqrt(rsq);
        if (domain->dimension == 3) {
          // Lucy Kernel, 3d
          // Note that wfd, the derivative of the weight function with respect to r,
          // is lacking a factor of r.
          // The missing factor of r is recovered by
          // (1) using delV . delX instead of delV . (delX/r) and
          // (2) using f[i][0] += delx * fpair instead of f[i][0] += (delx/r) * fpair
          wfd = -25.066903536973515383e0 * wfd * wfd * ihsq * ihsq * ihsq * ih;
        } else {
          // Lucy Kernel, 2d
          wfd = -19.098593171027440292e0 * wfd * wfd * ihsq * ihsq * ihsq;
        }

        // compute pressure  of atom j with Tait EOS
        tmp = rho[j] / rho0[jtype];
        fj = tmp * tmp * tmp;
        fj = B[jtype] * (fj * fj * tmp - 1.0) / (rho[j] * rho[j]);

        // dot product of velocity delta and distance vector
        delVdotDelR = delx * (vxtmp - v[j][0]) + dely * (vytmp - v[j][1])
            + delz * (vztmp - v[j][2]);

        // artificial viscosity (Monaghan 1992)

        if (delVdotDelR < 0.) {
        /************************************/
       
	mu = h * delVdotDelR / (rsq + 0.01 * h * h);
	
          
          fvisc = -viscosity[itype][jtype] * (soundspeed[itype]
              + soundspeed[jtype]) * mu / (rho[i] + rho[j]);
	//printf("fvisc : %f\n",fvisc);

	
        } else {
          fvisc = 0.;
        }

        // total pair force & thermal energy increment
        fpair = -imass * jmass * (fi + fj + fvisc) * wfd;
        deltaE = -0.5 * fpair * delVdotDelR;

        f[i][0] += delx * fpair;
        f[i][1] += dely * fpair;
        f[i][2] += delz * fpair;

        // and change in density
        drho[i] += jmass * delVdotDelR * wfd;

        // change in thermal energy
        de[i] += deltaE;

        

        if (evflag)
          ev_tally_full(i,0.0, 0.0, fpair, delx, dely, delz);
      }
    }
  }

  

	
	
}

