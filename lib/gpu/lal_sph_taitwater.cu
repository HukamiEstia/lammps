

#ifdef NV_KERNEL
#include "lal_aux_fun1.h"
#ifndef _DOUBLE_DOUBLE
texture<float4> pos_tex;
texture<float4> vel_tex;
#else
texture<int4,1> pos_tex;
texture<int4,1> vel_tex;
#endif
#else
#define pos_tex x_
#define vel_tex v_
#endif


__kernel void k_sph_taitwater(const __global numtyp4 *restrict x_,
const __global numtyp4 *restrict v_, 
                     const __global numtyp4 *restrict coeff,
                     const int lj_types,
                     const __global numtyp *restrict sp_lj_in,
                     const __global int *dev_nbor,
                     const __global int *dev_packed,
                     __global acctyp4 *restrict ans,
                     __global acctyp *restrict engv,
                     const int eflag, const int vflag, const int inum,
                     const int nbor_pitch, const int t_per_atom) {
  int tid, ii, offset;
  double imass,  tmp,fi,jmass,h, ih, ihsq,wfd,fj,delVdotDelR,mu,fvisc;
  double *rho,*rho0,*B,*soundspeed, *mass;
  atom_info(t_per_atom,ii,tid,offset);

  __local numtyp sp_lj[4];
  sp_lj[0]=sp_lj_in[0];
  sp_lj[1]=sp_lj_in[1];
  sp_lj[2]=sp_lj_in[2];
  sp_lj[3]=sp_lj_in[3];

  acctyp energy=(acctyp)0;
  acctyp4 f;
  f.x=(acctyp)0; f.y=(acctyp)0; f.z=(acctyp)0;
  acctyp virial[6];
  for (int i=0; i<6; i++)
    virial[i]=(acctyp)0;

  if (ii<inum) {
    int nbor, nbor_end;
    int i, numj;
    __local int n_stride;
    nbor_info(dev_nbor,dev_packed,nbor_pitch,t_per_atom,ii,offset,i,numj,
              n_stride,nbor_end,nbor);

    numtyp4 ix; fetch4(ix,i,pos_tex); //x_[i];
    int itype=ix.w;
	numtyp4 iv; fetch4(iv,i,vel_tex); //v_[i];
    int itag=iv.w;
    acctyp factor_lj;
	 imass = mass[itype];
	//compute pressure of atom i with Tait EOS
	tmp = rho[i]/rho0[itype];
	 fi = tmp * tmp * tmp;
	fi=B[itype] * (fi * fi * tmp - 1.0) / (rho[i]* rho[i]);
	
    for ( ; nbor<nbor_end; nbor+=n_stride) {

      int j=dev_packed[nbor];
      factor_lj = sp_lj[sbmask(j)];
      j &= NEIGHMASK;

      numtyp4 jx; fetch4(jx,j,pos_tex); //x_[j];
      int jtype=jx.w;
	  numtyp4 jv; fetch4(jv,j,vel_tex); //v_[j];
      int jtag=jv.w;

      // Compute r12
      numtyp delvx = iv.x-jv.x;
      numtyp delvy = iv.y-jv.y;
      numtyp delvz = iv.z-jv.z;
      numtyp rsq = delvx*delvx+delvy*delvy+delvz*delvz;
	  jmass = mass[jtype];
      int mtype=itype*lj_types+jtype;
      if (rsq<coeff[mtype].z) {
      
	   h = coeff[mtype].z;
	   ih = 1.0 / h;
       ihsq = ih * ih;
       wfd = h - sqrt(rsq);
        
       wfd = -19.098593171027440292e0 * wfd * wfd * ihsq * ihsq * ihsq;
		// compute pressure  of atom j with Tait EOS
        tmp = rho[j]/ rho[jtype];
        fj = tmp * tmp * tmp;
        fj = B[jtype]* (fj * fj * tmp - 1.0) / (rho[j] * rho[j]);
		// dot product of velocity delta and distance vector
        delVdotDelR = delvx * (iv.x - jv.x) + delvy * (iv.y - jv.y)
            + delvz * (iv.z - jv.z);
			if (delVdotDelR < 0.) {
          mu = h * delVdotDelR / (rsq + 0.01 * h * h);
          fvisc = -coeff[mtype].y * (soundspeed[itype]
              + soundspeed[jtype]) * mu / (rho[i] + rho[j]);
        } else {
          fvisc = 0.;
        }
			
			numtyp force = (numtyp)0.0;
			force=-imass * jmass * (fi + fj + fvisc) * wfd;
			numtyp deltaE = -0.5 * force * delVdotDelR;
			
        f.x+=delvx*force;
        f.y+=delvy*force;
        f.z+=delvz*force;

        if (eflag>0) {
          
          energy+=deltaE;
        }
        if (vflag>0) {
          virial[0] += delvx*delvx*force;
          virial[1] += delvy*delvy*force;
          virial[2] += delvz*delvz*force;
          virial[3] += delvx*delvy*force;
          virial[4] += delvx*delvz*force;
          virial[5] += delvy*delvz*force;
        }
      }

    } // for nbor
    store_answers(f,energy,virial,ii,inum,tid,t_per_atom,offset,eflag,vflag,
                  ans,engv);
  } // if ii
}

__kernel void k_sph_taitwater_fast(const __global numtyp4 *restrict x_,
                          const __global numtyp4 *restrict coeff_in,
                          const __global numtyp *restrict sp_lj_in,
                          const __global int *dev_nbor,
                          const __global int *dev_packed,
                          __global acctyp4 *restrict ans,
                          __global acctyp *restrict engv,
                          const int eflag, const int vflag, const int inum,
                          const int nbor_pitch, const int t_per_atom) {
  int tid, ii, offset;
  atom_info(t_per_atom,ii,tid,offset);

  __local numtyp4 coeff[MAX_SHARED_TYPES*MAX_SHARED_TYPES];
  __local numtyp sp_lj[4];
  if (tid<4)
    sp_lj[tid]=sp_lj_in[tid];
  if (tid<MAX_SHARED_TYPES*MAX_SHARED_TYPES) {
    coeff[tid]=coeff_in[tid];
  }

  acctyp energy=(acctyp)0;
  acctyp4 f;
  f.x=(acctyp)0; f.y=(acctyp)0; f.z=(acctyp)0;
  acctyp virial[6];
  for (int i=0; i<6; i++)
    virial[i]=(acctyp)0;

  __syncthreads();

  if (ii<inum) {
    int nbor, nbor_end;
    int i, numj;
    __local int n_stride;
    nbor_info(dev_nbor,dev_packed,nbor_pitch,t_per_atom,ii,offset,i,numj,
              n_stride,nbor_end,nbor);

    numtyp4 ix; fetch4(ix,i,pos_tex); //x_[i];
    int iw=ix.w;
    int itype=fast_mul((int)MAX_SHARED_TYPES,iw);

    numtyp factor_lj;
    for ( ; nbor<nbor_end; nbor+=n_stride) {

      int j=dev_packed[nbor];
      factor_lj = sp_lj[sbmask(j)];
      j &= NEIGHMASK;

      numtyp4 jx; fetch4(jx,j,pos_tex); //x_[j];
      int mtype=itype+jx.w;

      // Compute r12
      numtyp delx = ix.x-jx.x;
      numtyp dely = ix.y-jx.y;
      numtyp delz = ix.z-jx.z;
      numtyp rsq = delx*delx+dely*dely+delz*delz;

      if (rsq<coeff[mtype].z) {
        numtyp force;
        numtyp r = ucl_sqrt(rsq);
        numtyp arg = r/coeff[mtype].y;
        if (r > (numtyp)0.0) force = factor_lj * coeff[mtype].x *
                       sin(arg) *coeff[mtype].y*ucl_recip(r);
        else force = (numtyp)0.0;

        f.x+=delx*force;
        f.y+=dely*force;
        f.z+=delz*force;

        if (eflag>0) {
          numtyp e=coeff[mtype].x * ((numtyp)1.0+cos(arg));
          energy+=factor_lj*e;
        }
        if (vflag>0) {
          virial[0] += delx*delx*force;
          virial[1] += dely*dely*force;
          virial[2] += delz*delz*force;
          virial[3] += delx*dely*force;
          virial[4] += delx*delz*force;
          virial[5] += dely*delz*force;
        }
      }

    } // for nbor
    store_answers(f,energy,virial,ii,inum,tid,t_per_atom,offset,eflag,vflag,
                  ans,engv);
  } // if ii
}

