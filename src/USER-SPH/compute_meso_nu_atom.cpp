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

#include "compute_meso_nu_atom.h"
#include <cmath>
#include <cstring>
#include "atom.h"
#include "force.h"
#include "pair.h"
#include "pair_sph_taitwater.h"
#include "update.h"
#include "modify.h"
#include "comm.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeMesoNuAtom::ComputeMesoNuAtom(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg)
{
  if (narg != 3) error->all(FLERR,"Number of arguments for compute meso/nu/atom command != 3");
  if (atom->e_flag != 1) error->all(FLERR,"compute meso/nu/atom command requires atom_style with viscosity (e.g. meso)");

  peratom_flag = 1;
  size_peratom_cols = 0;

  nmax = 0;
  nuvector = NULL;
}

/* ---------------------------------------------------------------------- */

ComputeMesoNuAtom::~ComputeMesoNuAtom()
{
  memory->sfree(nuvector);
}

/* ---------------------------------------------------------------------- */

void ComputeMesoNuAtom::init()
{

  int count = 0;
  for (int i = 0; i < modify->ncompute; i++)
    if (strcmp(modify->compute[i]->style,"nuvector/atom") == 0) count++;
  if (count > 1 && comm->me == 0)
    error->warning(FLERR,"More than one compute nuvector/atom");
}

/* ---------------------------------------------------------------------- */

void ComputeMesoNuAtom::compute_peratom()
{
  invoked_peratom = update->ntimestep;

  // grow evector array if necessary

  if (atom->nmax > nmax) {
    memory->sfree(nuvector);
    nmax = atom->nmax;
    nuvector = (double *) memory->smalloc(nmax*sizeof(double),"nuvector/atom:nuvector");
    vector_atom = nuvector;
  }

  double *e = atom->e;
  double *cv = atom->cv;
  double *nu = atom->nu;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        double T = e[i]/cv[i];
        nuvector[i] = nu[i];
      }
      else {
        nuvector[i] = 0.0;
      }
    }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputeMesoNuAtom::memory_usage()
{
  double bytes = nmax * sizeof(double);
  return bytes;
}
