/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(sph/taitwater/andrade,PairSPHTaitwaterAndrade)

#else

#ifndef LMP_PAIR_TAITWATER_ANDRADE_H
#define LMP_PAIR_TAITWATER_ANDRADE_H

#include "pair.h"

namespace LAMMPS_NS {

class PairSPHTaitwaterAndrade : public Pair {
 public:
  PairSPHTaitwaterAndrade(class LAMMPS *);
  virtual ~PairSPHTaitwaterAndrade();
  virtual void compute(int, int);
  void settings(int, char **);
  void coeff(int, char **);
  virtual double init_one(int, int);

 protected:
  double *rho0, *soundspeed, *A, *B, *C;
  double **cut;
  int first;

  void allocate();
};

}

#endif
#endif
