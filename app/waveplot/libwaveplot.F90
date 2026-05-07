!--------------------------------------------------------------------------------------------------!
!  DFTB+: general package for performing fast atomistic simulations                                !
!  Copyright (C) 2006 - 2025  DFTB+ developers group                                               !
!                                                                                                  !
!  See the LICENSE file for terms of usage and distribution.                                       !
!--------------------------------------------------------------------------------------------------!
!
! C-bindable library interface for waveplot functionality.
! Allows Python (via ctypes) to evaluate molecular orbitals at arbitrary points.
! Pattern follows pyBall/Fireball/libFireCore.f90
!
! Array convention (Fortran column-major vs C/Python row-major):
!   Python points[npoints,3]  <->  Fortran points(3,npoints)
!   Python out[npoints]       <->  Fortran out(npoints)
!

#:include 'common.fypp'

!> Persistent state module - holds initialized data between calls
module libwaveplot_state
  use waveplot_slater,          only: TSlaterOrbital
  use waveplot_molorb,          only: TSpeciesBasis, TMolecularOrbital
  use dftbp_common_accuracy,    only: dp
  use dftbp_type_typegeometry,  only: TGeometry
  implicit none
  private

  type(TMolecularOrbital), allocatable, public, save :: molOrb
  type(TGeometry),                      public, save :: geo
  real(dp), allocatable,                public, save :: eigVecsReal(:,:)
  integer,                              public, save :: nOrbSystem = 0
  logical,                              public, save :: tInitialised = .false.

end module libwaveplot_state


!> C-bindable interface module — all bind(c) subroutines live here
module libwaveplot
  use iso_c_binding
  use dftbp_common_accuracy,    only: dp
  use libwaveplot_state
  use waveplot_molorb,          only: TSpeciesBasis, TMolecularOrbital, TMolecularOrbital_init, getValue
  use waveplot_slater,          only: TSlaterOrbital, TSlaterOrbital_init
  use dftbp_common_status,      only: TStatus
  use dftbp_dftb_boundarycond,  only: TBoundaryConds, TBoundaryConds_init, boundaryCondsEnum
  use dftbp_type_typegeometry,  only: TGeometry
  implicit none
  private

contains

  !> Initialize / reset library state. Call before any other function.
  subroutine waveplot_init() bind(c, name='waveplot_init')
    write(*,*) "waveplot_init()"
    tInitialised = .false.
    nOrbSystem   = 0
    if (allocated(molOrb))      deallocate(molOrb)
    if (allocated(eigVecsReal)) deallocate(eigVecsReal)
  end subroutine waveplot_init


  !> Set molecular geometry.
  !! coords_(3, natoms) in Fortran order  <->  Python coords[natoms,3] (pass .T)
  !! species_: 1-indexed species index per atom
  subroutine waveplot_set_geometry(natoms_, isPeriodic_, coords_, species_) &
      & bind(c, name='waveplot_set_geometry')
    integer(c_int), value, intent(in) :: natoms_, isPeriodic_
    real(c_double), intent(in)        :: coords_(3, natoms_)
    integer(c_int), intent(in)        :: species_(natoms_)
    integer :: ii
    write(*,*) "waveplot_set_geometry() natoms=", natoms_
    geo%nAtom              = natoms_
    geo%nSpecies           = maxval(species_)
    geo%tPeriodic          = (isPeriodic_ /= 0)
    geo%tFracCoord         = .false.
    geo%tHelical           = .false.
    geo%areContactsPresent = .false.
    if (allocated(geo%coords))       deallocate(geo%coords)
    if (allocated(geo%species))      deallocate(geo%species)
    if (allocated(geo%speciesNames)) deallocate(geo%speciesNames)
    allocate(geo%coords(3, natoms_))
    allocate(geo%species(natoms_))
    allocate(geo%speciesNames(geo%nSpecies))
    geo%coords(:,:)  = real(coords_(:,:), dp)
    geo%species(:)   = species_(:)
    do ii = 1, geo%nSpecies
      write(geo%speciesNames(ii), '(A,I0)') 'Sp', ii
    end do
    write(*,*) "DEBUG set_geometry: coords(Bohr):"
    do ii = 1, natoms_
      write(*,'(A,I3,A,3F12.6)') "  atom", ii, " sp=", geo%coords(:,ii)
    end do
    write(*,*) "DEBUG species:", geo%species(:)
  end subroutine waveplot_set_geometry


  !> Set STO basis for all species.
  !! Layout (all arrays in Fortran column-major order, Python passes .T):
  !!   nOrb_arr_(nSpecies)
  !!   angMom_arr_(nOrbMax, nSpecies)
  !!   cutoff_arr_(nOrbMax, nSpecies)
  !!   occ_arr_   (nOrbMax, nSpecies)
  !!   nAlpha_arr_(nOrbMax, nSpecies)
  !!   nPow_arr_  (nOrbMax, nSpecies)
  !!   alpha_flat_(nAlphaMax, nOrbMax, nSpecies)
  !!   aa_flat_   (nPowMax, nAlphaMax, nOrbMax, nSpecies)
  subroutine waveplot_set_basis(nSpecies_, nOrbMax_, nPowMax_, nAlphaMax_,  &
      & nOrb_arr_, angMom_arr_, cutoff_arr_, occ_arr_,                       &
      & nAlpha_arr_, nPow_arr_, alpha_flat_, aa_flat_, resolution_)           &
      & bind(c, name='waveplot_set_basis')
    integer(c_int), value, intent(in) :: nSpecies_, nOrbMax_, nPowMax_, nAlphaMax_
    integer(c_int), intent(in) :: nOrb_arr_(nSpecies_)
    integer(c_int), intent(in) :: angMom_arr_(nOrbMax_, nSpecies_)
    real(c_double), intent(in) :: cutoff_arr_(nOrbMax_, nSpecies_)
    real(c_double), intent(in) :: occ_arr_(nOrbMax_, nSpecies_)
    integer(c_int), intent(in) :: nAlpha_arr_(nOrbMax_, nSpecies_)
    integer(c_int), intent(in) :: nPow_arr_(nOrbMax_, nSpecies_)
    real(c_double), intent(in) :: alpha_flat_(nAlphaMax_, nOrbMax_, nSpecies_)
    real(c_double), intent(in) :: aa_flat_(nPowMax_, nAlphaMax_, nOrbMax_, nSpecies_)
    real(c_double), value, intent(in) :: resolution_
    ! locals
    type(TSpeciesBasis), allocatable :: basisArr(:)
    type(TBoundaryConds) :: boundaryCond
    type(TStatus) :: errStatus
    real(dp), allocatable :: aa_loc(:,:), alpha_loc(:)
    integer :: iSp, iOrb, nOrb, nA, nP

    write(*,*) "waveplot_set_basis() nSpecies=", nSpecies_
    allocate(basisArr(nSpecies_))
    do iSp = 1, nSpecies_
      nOrb = nOrb_arr_(iSp)
      basisArr(iSp)%nOrb        = nOrb
      basisArr(iSp)%atomicNumber = iSp
      allocate(basisArr(iSp)%angMoms    (nOrb))
      allocate(basisArr(iSp)%cutoffs    (nOrb))
      allocate(basisArr(iSp)%occupations(nOrb))
      allocate(basisArr(iSp)%stos       (nOrb))
      basisArr(iSp)%angMoms    (:) = angMom_arr_(1:nOrb, iSp)
      basisArr(iSp)%cutoffs    (:) = real(cutoff_arr_(1:nOrb, iSp), dp)
      basisArr(iSp)%occupations(:) = real(occ_arr_   (1:nOrb, iSp), dp)
      do iOrb = 1, nOrb
        nA = nAlpha_arr_(iOrb, iSp)
        nP = nPow_arr_  (iOrb, iSp)
        allocate(aa_loc(nP, nA))
        allocate(alpha_loc(nA))
        aa_loc(:,:)  = real(aa_flat_   (1:nP, 1:nA, iOrb, iSp), dp)
        alpha_loc(:) = real(alpha_flat_(1:nA, iOrb, iSp),        dp)
        call TSlaterOrbital_init(basisArr(iSp)%stos(iOrb), aa_loc, alpha_loc,     &
            & angMom_arr_(iOrb, iSp), real(resolution_, dp), real(cutoff_arr_(iOrb, iSp), dp))
        deallocate(aa_loc, alpha_loc)
      end do
    end do

    if (geo%tPeriodic) then
      call TBoundaryConds_init(boundaryCond, boundaryCondsEnum%pbc3d,  errStatus)
    else
      call TBoundaryConds_init(boundaryCond, boundaryCondsEnum%cluster, errStatus)
    end if

    write(*,*) "DEBUG set_basis: nSpecies=",nSpecies_," nOrbMax=",nOrbMax_
    do iSp = 1, nSpecies_
      write(*,'(A,I2,A,10I4)') "  sp",iSp," angMoms:", angMom_arr_(1:nOrb_arr_(iSp),iSp)
      write(*,'(A,I2,A,10F8.3)') "  sp",iSp," cutoffs:", cutoff_arr_(1:nOrb_arr_(iSp),iSp)
      write(*,'(A,I2,A,10I4)') "  sp",iSp," nAlpha:", nAlpha_arr_(1:nOrb_arr_(iSp),iSp)
      do iOrb = 1, nOrb_arr_(iSp)
        nA = nAlpha_arr_(iOrb,iSp)
        write(*,'(A,I2,A,10F8.4)') "    orb",iOrb," alpha:", alpha_flat_(1:nA,iOrb,iSp)
      end do
    end do
    if (allocated(molOrb)) deallocate(molOrb)
    allocate(molOrb)
    call TMolecularOrbital_init(molOrb, geo, boundaryCond, basisArr)
    tInitialised = .true.
    write(*,*) "waveplot_set_basis() done, nOrbSystem=", nOrbSystem
  end subroutine waveplot_set_basis


  !> Set real-space eigenvectors.
  !! eigvecs_(nOrb, nStates) in Fortran order  <->  Python eigvecs[nStates, nOrb] (pass .T)
  subroutine waveplot_set_eigenvectors(nOrb_, nStates_, eigvecs_) &
      & bind(c, name='waveplot_set_eigenvectors')
    integer(c_int), value, intent(in) :: nOrb_, nStates_
    real(c_double), intent(in)        :: eigvecs_(nOrb_, nStates_)
    write(*,*) "waveplot_set_eigenvectors() nOrb=", nOrb_, " nStates=", nStates_
    if (allocated(eigVecsReal)) deallocate(eigVecsReal)
    allocate(eigVecsReal(nOrb_, nStates_))
    eigVecsReal(:,:) = real(eigvecs_(:,:), dp)
    nOrbSystem = nOrb_
    write(*,*) "DEBUG eigvecs MO1:", eigVecsReal(:,1)
    write(*,*) "DEBUG eigvecs MO4:", eigVecsReal(:,4)
  end subroutine waveplot_set_eigenvectors


  !> Evaluate one MO (iState, 1-indexed) at arbitrary points.
  !! points_(3, npoints) Fortran  <->  Python points[npoints,3] (pass .T or np.asfortranarray)
  !! out_(npoints): wavefunction values
  subroutine waveplot_orb2points(iState_, npoints_, points_, out_) &
      & bind(c, name='waveplot_orb2points')
    integer(c_int), value, intent(in) :: iState_, npoints_
    real(c_double), intent(in)        :: points_(3, npoints_)
    real(c_double), intent(out)       :: out_(npoints_)
    real(dp) :: origin(3), gridVecs(3,3)
    real(dp), allocatable :: valueOnGrid(:,:,:,:)
    integer :: ip

    if (.not. tInitialised) then
      write(*,*) "ERROR waveplot_orb2points: not initialised"
      out_(:) = 0.0_dp; return
    end if
    if (iState_ < 1 .or. iState_ > size(eigVecsReal, dim=2)) then
      write(*,*) "ERROR waveplot_orb2points: iState out of range", iState_
      out_(:) = 0.0_dp; return
    end if

    allocate(valueOnGrid(1, 1, 1, 1))
    gridVecs(:,:) = 0.0_dp
    do ip = 1, npoints_
      origin(:) = real(points_(:, ip), dp)
      call getValue(molOrb, origin, gridVecs, eigVecsReal(:, iState_:iState_), valueOnGrid)
      out_(ip) = valueOnGrid(1, 1, 1, 1)
    end do
    deallocate(valueOnGrid)
  end subroutine waveplot_orb2points


  !> Evaluate all nStates MOs at a single point.
  !! out_(nStates): wavefunction values for all states at the given point.
  subroutine waveplot_allorbs2point(point_, nStates_, out_) &
      & bind(c, name='waveplot_allorbs2point')
    real(c_double), intent(in)        :: point_(3)
    integer(c_int), value, intent(in) :: nStates_
    real(c_double), intent(out)       :: out_(nStates_)
    real(dp) :: origin(3), gridVecs(3,3)
    real(dp), allocatable :: valueOnGrid(:,:,:,:)

    if (.not. tInitialised) then
      write(*,*) "ERROR waveplot_allorbs2point: not initialised"
      out_(:) = 0.0_dp; return
    end if

    allocate(valueOnGrid(1, 1, 1, nStates_))
    gridVecs(:,:) = 0.0_dp
    origin(:) = real(point_(:), dp)
    call getValue(molOrb, origin, gridVecs, eigVecsReal(:, 1:nStates_), valueOnGrid)
    out_(:) = real(valueOnGrid(1, 1, 1, :), c_double)
    deallocate(valueOnGrid)
  end subroutine waveplot_allorbs2point


  !> Evaluate one MO on a regular 3D grid.
  !! origin_(3): grid origin in Bohr
  !! gridVecs_(3,3): step vectors (columns), Fortran order  <->  Python gridVecs[3,3] (pass .T)
  !! nPoints_(3): number of points along each direction
  !! out_(n1*n2*n3): flattened output in Fortran column-major order
  subroutine waveplot_orb2grid(iState_, origin_, gridVecs_, nPoints_, out_) &
      & bind(c, name='waveplot_orb2grid')
    integer(c_int), value, intent(in) :: iState_
    real(c_double), intent(in)        :: origin_(3)
    real(c_double), intent(in)        :: gridVecs_(3,3)
    integer(c_int), intent(in)        :: nPoints_(3)
    real(c_double), intent(out)       :: out_(nPoints_(1)*nPoints_(2)*nPoints_(3))
    real(dp), allocatable :: valueOnGrid(:,:,:,:)
    real(dp) :: orig(3), gv(3,3)
    integer :: n1, n2, n3

    if (.not. tInitialised) then
      write(*,*) "ERROR waveplot_orb2grid: not initialised"
      out_(:) = 0.0_dp; return
    end if

    n1 = nPoints_(1); n2 = nPoints_(2); n3 = nPoints_(3)
    allocate(valueOnGrid(n1, n2, n3, 1))
    orig(:) = real(origin_(:),      dp)
    gv(:,:) = real(gridVecs_(:,:),  dp)
    write(*,'(A,I3,A,3I4)') "DEBUG orb2grid iState=",iState_," nPoints=",n1,n2,n3
    write(*,'(A,3F10.5)') "  origin(Bohr):", orig
    write(*,'(A,3F10.5)') "  gv col1(dA):", gv(:,1)
    write(*,'(A,3F10.5)') "  gv col2(dB):", gv(:,2)
    write(*,'(A,3F10.5)') "  gv col3(dC):", gv(:,3)
    call getValue(molOrb, orig, gv, eigVecsReal(:, iState_:iState_), valueOnGrid)
    out_(:) = real(reshape(valueOnGrid(:,:,:,1), [n1*n2*n3]), c_double)
    deallocate(valueOnGrid)
  end subroutine waveplot_orb2grid


  !> Return number of basis orbitals in the system (-1 if not ready).
  subroutine waveplot_get_nOrb(nOrb_) bind(c, name='waveplot_get_nOrb')
    integer(c_int), intent(out) :: nOrb_
    if (tInitialised .and. allocated(eigVecsReal)) then
      nOrb_ = nOrbSystem
    else
      nOrb_ = -1
    end if
  end subroutine waveplot_get_nOrb

end module libwaveplot
