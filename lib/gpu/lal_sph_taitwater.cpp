
#ifdef USE_OPENCL
#include "sph_taitwater_cl.h"
#elif defined(USE_CUDART)
const char *sph_taitwater=0;
#else
#include "sph_taitwater_cubin.h"
#endif




#include "lal_sph_taitwater.h"
#include <cassert>
namespace LAMMPS_AL {
#define SphTaitwaterT Sph_taitwater<numtyp, acctyp>

extern Device<PRECISION,ACC_PRECISION> device;

template <class numtyp, class acctyp>
SphTaitwaterT::Sph_taitwater() : BaseAtomicsph<numtyp,acctyp>(), _allocated(false) {
}

template <class numtyp, class acctyp>
SphTaitwaterT::~Sph_taitwater() {
  clear();
}

template <class numtyp, class acctyp>
int SphTaitwaterT::bytes_per_atom(const int max_nbors) const {
  return this->bytes_per_atom_atomic(max_nbors);
}

template <class numtyp, class acctyp>
int SphTaitwaterT::init(const int ntypes, double **host_cutsq,double *host_B,
                 double **host_viscosity,double *host_rho, double **host_cut,
                 double *host_soundspeed,double *host_special_lj, const int nlocal,
                 const int nall, const int max_nbors,
                 const int maxspecial, const double cell_size,
                 const double gpu_split, FILE *_screen) {
  int success;
  success=this->init_atomic(nlocal,nall,max_nbors,maxspecial,cell_size,gpu_split,
                            _screen,sph_taitwater,"k_sph_taitwater");
	printf("success: %f\n", success);
  if (success!=0)
    return success;

  // If atom type constants fit in shared memory use fast kernel
  int lj_types=ntypes;
  shared_types=false;
  int max_shared_types=this->device->max_shared_types();
  
  printf("shared_types: %f\n", shared_types);
  
  if (lj_types<=max_shared_types && this->_block_size>=max_shared_types) {
    lj_types=max_shared_types;
    shared_types=true;
  }
  _lj_types=lj_types;

  printf("lj_types: %f\n", lj_types);
  
  // Allocate a host write buffer for data initialization
  UCL_H_Vec<numtyp> host_write(lj_types*lj_types*32,*(this->ucl_device),
                               UCL_WRITE_ONLY);

  for (int i=0; i<lj_types*lj_types; i++)
    host_write[i]=0.0;

  coeff.alloc(lj_types*lj_types,*(this->ucl_device),UCL_READ_ONLY);
  this->atom->type_pack4(ntypes,lj_types,coeff,host_write,host_cut,host_viscosity,host_cutsq);

  
  UCL_H_Vec<double> dview;
  sp_lj.alloc(4,*(this->ucl_device),UCL_READ_ONLY);
  dview.view(host_special_lj,4,*(this->ucl_device));
  ucl_copy(sp_lj,dview,false);

  _allocated=true;
  this->_max_bytes=coeff.row_bytes()+sp_lj.row_bytes();
  return 0;
}


template <class numtyp, class acctyp>
void SphTaitwaterT::reinit(const int ntypes, double **host_cutsq,double **host_viscosity,double **host_cut ) {

  // Allocate a host write buffer for data initialization
  UCL_H_Vec<numtyp> host_write(_lj_types*_lj_types*32,*(this->ucl_device),
                               UCL_WRITE_ONLY);

  for (int i=0; i<_lj_types*_lj_types; i++)
    host_write[i]=0.0;

  this->atom->type_pack4(ntypes,_lj_types,coeff,host_write,host_cut,host_viscosity,host_cutsq);
}

template <class numtyp, class acctyp>
void SphTaitwaterT::clear() {
  if (!_allocated)
    return;
  _allocated=false;

  coeff.clear();
  sp_lj.clear();
  this->clear_atomic();
}

template <class numtyp, class acctyp>
double SphTaitwaterT::host_memory_usage() const {
  return this->host_memory_usage_atomic()+sizeof(Sph_taitwater<numtyp,acctyp>);
}

// ---------------------------------------------------------------------------
// Calculate energies, forces, and torques
// ---------------------------------------------------------------------------
template <class numtyp, class acctyp>
void SphTaitwaterT::loop(const bool _eflag, const bool _vflag) {
  // Compute the block size and grid size to keep all cores busy
  const int BX=this->block_size();
  int eflag, vflag;
  if (_eflag)
    eflag=1;
  else
    eflag=0;

  if (_vflag)
    vflag=1;
  else
    vflag=0;

  int GX=static_cast<int>(ceil(static_cast<double>(this->ans->inum())/
                               (BX/this->_threads_per_atom)));

  int ainum=this->ans->inum();
  int nbor_pitch=this->nbor->nbor_pitch();
  this->time_pair.start();
  if (shared_types) {
    this->k_pair_fast.set_size(GX,BX);
    this->k_pair_fast.run(&this->atom->x,&coeff, &sp_lj,
                          &this->nbor->dev_nbor, &this->_nbor_data->begin(),
                          &this->ans->force, &this->ans->engv, &eflag, &vflag,
                          &ainum, &nbor_pitch, &this->_threads_per_atom);
  } else {
    this->k_pair.set_size(GX,BX);
    this->k_pair.run(&this->atom->x,&this->atom->v,&coeff, &_lj_types, &sp_lj,
                     &this->nbor->dev_nbor, &this->_nbor_data->begin(),
                     &this->ans->force, &this->ans->engv, &eflag, &vflag,
                     &ainum, &nbor_pitch, &this->_threads_per_atom);
  }
  this->time_pair.stop();
}

template class Sph_taitwater<PRECISION,ACC_PRECISION>;
}
