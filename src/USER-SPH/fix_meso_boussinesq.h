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

#ifdef FIX_CLASS

FixStyle(meso/boussinesq,FixMesoBoussinesq)

#else

#ifndef LMP_FIX_MESO_BOUSSINESQ_H
#define LMP_FIX_MESO_BOUSSINESQ_H

#include "fix.h"

namespace LAMMPS_NS {

class FixMesoBoussinesq : public Fix {
  friend class FixPour;

 public:
  FixMesoBoussinesq(class LAMMPS *, int, char **);
  virtual ~FixMesoBoussinesq();
  int setmask();
  void init();
  void setup(int);
  virtual void post_force(int);
  virtual void post_force_respa(int, int, int);
  double compute_scalar();
  void *extract(const char *, int &);

 protected:
  int disable;
  double magnitude;
  double alpha;
  double xdir,ydir,zdir;
  double xgrav,ygrav,zgrav,xacc,yacc,zacc;
  double degree2rad; // need??
  int ilevel_respa;
  int time_origin;
  double gvec[3];

  int eflag;
  double egrav,egrav_all;

  int varflag;
  int mstyle,astyle,xstyle,ystyle,zstyle;
  int mvar,avar,xvar,yvar,zvar;
  char *mstr,*astr,*xstr,*ystr,*zstr;

  void set_acceleration();
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Variable name for fix gravity does not exist

Self-explanatory.

E: Variable for fix gravity is invalid style

Only equal-style variables can be used.

*/
