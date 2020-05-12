/* ----------------------------------------------------------------------
 LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
 http://lammps.sandia.gov, Sandia National Laboratories
 Steve Plimpton, sjplimp@sandia.gov

 Copyright (2003) Sandia Corporation.  Under the terms of Contract
 DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
 certain rights in this software.  This software is distributed under
 the GNU General Public License.

 See the README file in the top-level LAMMPS directory.
 ------------------------------------------------------------------------- */

#include "pair_sph_lj_woodcock.h"
#include <cmath>
#include "atom.h"
#include "force.h"
#include "neigh_list.h"
#include "memory.h"
#include "error.h"
#include "domain.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairSPHLJWoodcock::PairSPHLJWoodcock(LAMMPS *lmp) : Pair(lmp)
{
  restartinfo = 0;
}

/* ---------------------------------------------------------------------- */

PairSPHLJWoodcock::~PairSPHLJWoodcock() {
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(cut);
    memory->destroy(eparam);
    memory->destroy(lparam);
    memory->destroy(nu0);
    memory->destroy(C);
  }
}

/* ---------------------------------------------------------------------- */

void PairSPHLJWoodcock::compute(int eflag, int vflag) {
  int i, j, ii, jj, inum, jnum, itype, jtype;
  double xtmp, ytmp, ztmp, delx, dely, delz, fpair;

  int *ilist, *jlist, *numneigh, **firstneigh;
  double vxtmp, vytmp, vztmp, r, rsq, wf, wfd, delVdotDelR, h, ih, ihsq, ihcub, lrc;
  double imass, jmass, mu, fi, fj, fvisc, ivisc, jvisc, ijvisc, Ti, Tj, deltaE, ci, cj;

  ev_init(eflag, vflag);

  double **v = atom->vest;
  double **x = atom->x;
  double **f = atom->f;
  double *rho = atom->rho;
  double *mass = atom->mass;
  double *de = atom->de;
  double *e = atom->e;
  double *cv = atom->cv;
  double *drho = atom->drho;
  double *nu = atom->nu;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

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

    // compute pressure, soundspeed and viscosity of particle i with LJ EOS
    LJEOS2(rho[i], e[i], cv[i], eparam[itype], lparam[itype], &fi, &ci);
    fi /= (rho[i] * rho[i]);
    //printf("fi = %f\n", fi);
    LJvisc(rho[i], e[i], cv[i], eparam[itype], lparam[itype], nu0[itype], C[itype], &nu[i]);

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
        ihcub = ihsq * ih;

        wfd = h - sqrt(rsq);
        if (domain->dimension == 3) {
          // Lucy Kernel, 3d
          // Note that wfd, the derivative of the weight function with respect to r,
          // is lacking a factor of r.
          // The missing factor of r is recovered by
          // (1) using delV . delX instead of delV . (delX/r) and
          // (2) using f[i][0] += delx * fpair instead of f[i][0] += (delx/r) * fpair
          wf =  2.0889086280811262819e0 * (h + 3. * r) * wf * wf * wf * ih;
          wfd = -25.066903536973515383e0 * wfd * wfd * ihsq * ihsq * ihsq * ih;
        } else {
          // Lucy Kernel, 2d
          wf = 1.5915494309189533576e0 * (h + 3. * r) * wf * wf * wf;
          wfd = -19.098593171027440292e0 * wfd * wfd * ihsq * ihsq * ihsq;
        }

        // compute pressure, soundspeed and viscosity of particle j with LJ EOS
        // LJEOS2(rho[j], e[j], cv[j], &fj, &cj);
        fj /= (rho[j] * rho[j]);

        // apply long-range correction to model a LJ fluid with cutoff
        // this implies that the modelled LJ fluid has cutoff == SPH cutoff
        lrc = - 11.1701 * (ihcub * ihcub * ihcub - 1.5 * ihcub);
        fi += lrc;
        fj += lrc;

        // dot product of velocity delta and distance vector
        delVdotDelR = delx * (vxtmp - v[j][0]) + dely * (vytmp - v[j][1])
            + delz * (vztmp - v[j][2]);

        // averaged artificial viscosities
        if (delVdotDelR < 0.) {
          mu = h * delVdotDelR / (rsq + 0.01 * h * h);
          ivisc = 8 * nu[i] / (rho[i] * ci * h);
          jvisc = 8 * nu[j] / (rho[j] * cj * h);
          ijvisc = wf * (imass * ivisc + jmass * jvisc) / (rho[i] + rho[j]);
          fvisc = -ijvisc * (ci + cj) * mu / (rho[i] + rho[j]);
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

        if (newton_pair || j < nlocal) {
          f[j][0] -= delx * fpair;
          f[j][1] -= dely * fpair;
          f[j][2] -= delz * fpair;
          de[j] += deltaE;
          drho[j] += imass * delVdotDelR * wfd;
        }

        if (evflag)
          ev_tally(i, j, nlocal, newton_pair, 0.0, 0.0, fpair, delx, dely, delz);
      }
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
 allocate all arrays
 ------------------------------------------------------------------------- */

void PairSPHLJWoodcock::allocate() {
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");

  memory->create(cut, n + 1, n + 1, "pair:cut");
  memory->create(eparam, n + 1, "pair:eparam");
  memory->create(lparam, n + 1, "pair:lparam");
  memory->create(nu0, n + 1, "pair:nu0");
  memory->create(C, n + 1, "pair:C");

}

/* ----------------------------------------------------------------------
 global settings
 ------------------------------------------------------------------------- */

void PairSPHLJWoodcock::settings(int narg, char **/*arg*/) {
  if (narg != 0)
    error->all(FLERR,
        "Illegal number of arguments for pair_style sph/lj/woodcock");
}

/* ----------------------------------------------------------------------
 set coeffs for one or more type pairs
 ------------------------------------------------------------------------- */

void PairSPHLJWoodcock::coeff(int narg, char **arg) {
  if (narg != 7)
    error->all(FLERR,
        "Incorrect args for pair_style sph/lj/woodcock coefficients");
  if (!allocated)
    allocate();

  int ilo, ihi, jlo, jhi;
  force->bounds(FLERR,arg[0], atom->ntypes, ilo, ihi);
  force->bounds(FLERR,arg[1], atom->ntypes, jlo, jhi);

  double eparam_one = force->numeric(FLERR,arg[2]);
  double lparam_one = force->numeric(FLERR,arg[3]);
  double nu0_one = force->numeric(FLERR,arg[4]);
  double C_one = force->numeric(FLERR,arg[5]);
  double cut_one = force->numeric(FLERR,arg[6]);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    eparam[i] = eparam_one;
    lparam[i] = lparam_one;
    nu0[i] = nu0_one;
    C[i] = C_one;
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      printf("setting cut[%d][%d] = %f\n", i, j, cut_one);
      cut[i][j] = cut_one;
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0)
    error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
 init for one type pair i,j and corresponding j,i
 ------------------------------------------------------------------------- */

double PairSPHLJWoodcock::init_one(int i, int j) {

  if (setflag[i][j] == 0) {
    error->all(FLERR,"All pair sph/lj coeffs are not set");
  }

  cut[j][i] = cut[i][j];

  return cut[i][j];
}

/* ---------------------------------------------------------------------- */

double PairSPHLJWoodcock::single(int /*i*/, int /*j*/, int /*itype*/, int /*jtype*/,
    double /*rsq*/, double /*factor_coul*/, double /*factor_lj*/, double &fforce) {
  fforce = 0.0;

  return 0.0;
}


/* --------------------------------------------------------------------------------------------- */
/* Lennard-Jones EOS,
   Francis H. Ree
   "Analytic representation of thermodynamic data for the Lennard‐Jones fluid",
   Journal of Chemical Physics 73 pp. 5401-5403 (1980)
*/

void PairSPHLJWoodcock::LJEOS2(double rho, double e, double cv,
                               double eparam, double lparam,
                               double *p, double *c) {
  double T = e/cv;
  double T_star = 1.38064852e-23 * T / eparam;
  double rho_star = rho * lparam * lparam * lparam;
  double beta = 1.0 / 1.38064852e-23 * T;
  double invTs = 1.0 / T_star;
  double invTs_sqrt = sqrt(invTs);
  double x = rho_star * sqrt(invTs_sqrt);

  double xsq = x * x;
  double xpow3 = xsq * x;
  double xpow4 = xsq * xsq;

  /* differential of Helmholtz free energy w.r.t. x */
  double diff_A_NkT = 3.629 + 7.264*x - beta*(3.492 - 18.698*x + 35.505*xsq - 31.816*xpow3 + 11.195*xpow4)
                    - invTs_sqrt*(5.369 + 13.16*x + 18.525*xsq - 17.076*xpow3 + 9.32*xpow4)
                    + 10.4925*xsq + 11.46*xpow3 + 2.176*xpow4*xpow4*x;

 /* differential of Helmholtz free energy w.r.t. x^2 */
  double d2A_dx2 = 7.264 + 20.985*x \
                 + beta*(18.698 - 71.01*x + 95.448*xsq - 44.78*xpow3)\
                 - invTs_sqrt*(13.16 + 37.05*x - 51.228*xsq + 37.28*xpow3)\
                 + 34.38*xsq + 19.584*xpow4*xpow4;

  // p = rho k T * (1 + rho * d(A/(NkT))/drho)
  // dx/drho = rho/x
  *p = rho * T * (1.0 + diff_A_NkT * x); // pressure
  double csq = T * (1.0 + 2.0 * diff_A_NkT * x + d2A_dx2 * x * x); // soundspeed squared
  if (csq > 0.0) {
    *c = sqrt(csq); // soundspeed
  } else {
    *c = 0.0;
  }
}


void PairSPHLJWoodcock::LJvisc(double rho, double e, double cv, 
                               double eparam, double lparam, 
                               double nu0, double C, double *nu) {
  double T = e/cv;
  double T_star = 1.38064852e-23 * T / eparam;
  double rho_star = rho * lparam * lparam * lparam;
  double invTs = 1.0 / T_star;
  double invTs_cbrt = sqrt(invTs);
  double Tspow4 = Tspow4 * Tspow4 * Tspow4 * Tspow4;
  double rspow4 = rho_star * rho_star * rho_star * rho_star;

  double Bn = sqrt(2) * (1 - T_star/8 - 1/Tspow4);

  *nu = nu0 * T_star * (1 + Bn * rho_star + C * invTs_cbrt * rspow4);

}