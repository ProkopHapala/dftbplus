!> Module to store H, S, DM (dense) at well-defined points inside the SCF loop
module dftbp_dftbplus_hamiltonian_store
  use dftbp_common_accuracy, only: dp
  implicit none
  private

  public :: set_store_hamiltonian, store_hamiltonian, get_stored_hamiltonian
  public :: store_overlap, get_stored_overlap
  public :: store_dm, get_stored_dm
  public :: store_eigvecs, get_stored_eigvecs
  public :: clear_stored_matrices
  public :: store_hamiltonian_cplx, store_overlap_cplx, store_dm_cplx, store_eigvecs_cplx
  public :: get_stored_hamiltonian_cplx, get_stored_overlap_cplx, get_stored_dm_cplx
  public :: get_stored_eigvecs_cplx, get_cplx_store_dims

  real(dp), allocatable, save :: storedH(:,:)
  real(dp), allocatable, save :: storedS(:,:)
  real(dp), allocatable, save :: storedDM(:,:)
  real(dp), allocatable, save :: storedEigvecs(:,:)   ! (norb, nstates) for iKS=1, iSpin=1
  real(dp), allocatable, save :: storedEigenvals(:)   ! (nstates) for iKS=1, iSpin=1

  ! Complex (k-point) storage. Slot index iKS = iK + (iSpin-1)*nKPoint (global (k,s) composite).
  complex(dp), allocatable, save :: storedHCplx(:,:,:)      ! (norb, norb, iKS)
  complex(dp), allocatable, save :: storedSCplx(:,:,:)
  complex(dp), allocatable, save :: storedDMCplx(:,:,:)
  complex(dp), allocatable, save :: storedEigvecsCplx(:,:,:) ! (norb, nstates, iKS)
  real(dp),    allocatable, save :: storedEigvalsCplx(:,:)   ! (nstates, iKS)
  logical,     allocatable, save :: cplxSlotFilled(:)
  integer, save :: nOrbCplx = 0, nKScplx = 0

  logical,  save :: tStoreMatrices = .false.
  integer,  save :: storedSize = 0

contains

  subroutine set_store_hamiltonian(store)
    logical, intent(in) :: store
    tStoreMatrices = store
  end subroutine

  subroutine store_hamiltonian(H, sizeH)
    real(dp), intent(in) :: H(:,:)
    integer,  intent(in) :: sizeH
    integer :: i, j
    if (.not. tStoreMatrices) return
    if (allocated(storedH)) deallocate(storedH)
    allocate(storedH(sizeH, sizeH), source=0.0_dp)
    do j = 1, sizeH  ! H from unpackHS: lower triangle filled
      do i = j, sizeH
        storedH(i,j) = H(i,j)
        storedH(j,i) = H(i,j)
      end do
    end do
    storedSize = sizeH
  end subroutine

  subroutine store_overlap(S, sizeS)
    real(dp), intent(in) :: S(:,:)
    integer,  intent(in) :: sizeS
    integer :: i, j
    if (.not. tStoreMatrices) return
    if (allocated(storedS)) deallocate(storedS)
    allocate(storedS(sizeS, sizeS), source=0.0_dp)
    do j = 1, sizeS  ! S from unpackHS: lower triangle filled
      do i = j, sizeS
        storedS(i,j) = S(i,j)
        storedS(j,i) = S(i,j)
      end do
    end do
  end subroutine

  subroutine store_dm(DM, sizeDM)
    real(dp), intent(in) :: DM(:,:)
    integer,  intent(in) :: sizeDM
    integer :: i, j
    if (.not. tStoreMatrices) return
    if (allocated(storedDM)) deallocate(storedDM)
    allocate(storedDM(sizeDM, sizeDM), source=0.0_dp)
    ! herk fills only lower triangle -- copy lower and mirror to upper
    do j = 1, sizeDM
      do i = j, sizeDM  ! i >= j : lower triangle
        storedDM(i,j) = DM(i,j)
        storedDM(j,i) = DM(i,j)
      end do
    end do
  end subroutine

  subroutine get_stored_hamiltonian(H, sizeH)
    real(dp), intent(out) :: H(:,:)
    integer,  intent(out) :: sizeH
    if (allocated(storedH)) then; H = storedH; sizeH = storedSize
    else;                          H = 0.0_dp;  sizeH = 0
    end if
  end subroutine

  subroutine get_stored_overlap(S, sizeS)
    real(dp), intent(out) :: S(:,:)
    integer,  intent(out) :: sizeS
    if (allocated(storedS)) then; S = storedS; sizeS = storedSize
    else;                          S = 0.0_dp;  sizeS = 0
    end if
  end subroutine

  subroutine get_stored_dm(DM, sizeDM)
    real(dp), intent(out) :: DM(:,:)
    integer,  intent(out) :: sizeDM
    if (allocated(storedDM)) then; DM = storedDM; sizeDM = storedSize
    else;                           DM = 0.0_dp;   sizeDM = 0
    end if
  end subroutine

  !> Store eigenvectors and eigenvalues for iKS=1 after diagonalization.
  !> eigvecs_in: (norb, norb) where columns are MOs (Fortran convention after diagDenseMtx)
  !> eigenvals_in: (norb) eigenvalues
  subroutine store_eigvecs(eigvecs_in, eigenvals_in, norb)
    real(dp), intent(in) :: eigvecs_in(:,:)
    real(dp), intent(in) :: eigenvals_in(:)
    integer,  intent(in) :: norb
    if (.not. tStoreMatrices) return
    if (allocated(storedEigvecs))  deallocate(storedEigvecs)
    if (allocated(storedEigenvals)) deallocate(storedEigenvals)
    allocate(storedEigvecs(norb, norb), source=0.0_dp)
    allocate(storedEigenvals(norb), source=0.0_dp)
    storedEigvecs  = eigvecs_in(1:norb, 1:norb)
    storedEigenvals = eigenvals_in(1:norb)
    storedSize = norb
  end subroutine

  subroutine get_stored_eigvecs(eigvecs_out, eigenvals_out, norb)
    real(dp), intent(out) :: eigvecs_out(:,:)
    real(dp), intent(out) :: eigenvals_out(:)
    integer,  intent(out) :: norb
    if (allocated(storedEigvecs)) then
      norb = storedSize
      eigvecs_out = storedEigvecs
      eigenvals_out = storedEigenvals
    else
      norb = 0
      eigvecs_out  = 0.0_dp
      eigenvals_out = 0.0_dp
    end if
  end subroutine

  ! ---- Complex (k-point) storage ----
  ! Slot convention: iKS = iK + (iSpin-1)*nKPoint, matching (iS-1)*nKpoint + iK in
  ! TParallelKS_init. All matrices stored fully Hermitian (lower triangle from
  ! unpackHS/herk is mirrored to the upper triangle with conjugation).

  !> Ensure complex storage arrays are allocated with the right dimensions.
  subroutine ensure_cplx_store(nOrb, nKS)
    integer, intent(in) :: nOrb, nKS
    if (nOrbCplx == nOrb .and. nKScplx == nKS .and. allocated(storedHCplx)) return
    if (allocated(storedHCplx))       deallocate(storedHCplx)
    if (allocated(storedSCplx))       deallocate(storedSCplx)
    if (allocated(storedDMCplx))      deallocate(storedDMCplx)
    if (allocated(storedEigvecsCplx)) deallocate(storedEigvecsCplx)
    if (allocated(storedEigvalsCplx)) deallocate(storedEigvalsCplx)
    if (allocated(cplxSlotFilled))    deallocate(cplxSlotFilled)
    allocate(storedHCplx(nOrb, nOrb, nKS),       source=cmplx(0.0_dp, 0.0_dp, dp))
    allocate(storedSCplx(nOrb, nOrb, nKS),       source=cmplx(0.0_dp, 0.0_dp, dp))
    allocate(storedDMCplx(nOrb, nOrb, nKS),      source=cmplx(0.0_dp, 0.0_dp, dp))
    allocate(storedEigvecsCplx(nOrb, nOrb, nKS), source=cmplx(0.0_dp, 0.0_dp, dp))
    allocate(storedEigvalsCplx(nOrb, nKS),       source=0.0_dp)
    allocate(cplxSlotFilled(nKS),                source=.false.)
    nOrbCplx = nOrb
    nKScplx = nKS
  end subroutine ensure_cplx_store

  !> Store complex Hamiltonian H(k) for (iK, iSpin); lower triangle from unpackHS is
  !> mirrored to upper with conjugation (Hermitian).
  subroutine store_hamiltonian_cplx(H, iK, iSpin, nK, nSpin)
    complex(dp), intent(in) :: H(:,:)
    integer, intent(in) :: iK, iSpin, nK, nSpin
    integer :: i, j, iKS, n
    if (.not. tStoreMatrices) return
    n = size(H, dim=1)
    call ensure_cplx_store(n, nK * nSpin)
    iKS = iK + (iSpin - 1) * nK
    do j = 1, n
      do i = j, n
        storedHCplx(i,j,iKS) = H(i,j)
        storedHCplx(j,i,iKS) = conjg(H(i,j))
      end do
    end do
    cplxSlotFilled(iKS) = .true.
  end subroutine store_hamiltonian_cplx

  subroutine store_overlap_cplx(S, iK, iSpin, nK, nSpin)
    complex(dp), intent(in) :: S(:,:)
    integer, intent(in) :: iK, iSpin, nK, nSpin
    integer :: i, j, iKS, n
    if (.not. tStoreMatrices) return
    n = size(S, dim=1)
    call ensure_cplx_store(n, nK * nSpin)
    iKS = iK + (iSpin - 1) * nK
    do j = 1, n
      do i = j, n
        storedSCplx(i,j,iKS) = S(i,j)
        storedSCplx(j,i,iKS) = conjg(S(i,j))
      end do
    end do
  end subroutine store_overlap_cplx

  !> Store complex k-space density matrix (lower triangle from herk, mirrored Hermitian).
  subroutine store_dm_cplx(DM, iK, iSpin, nK, nSpin)
    complex(dp), intent(in) :: DM(:,:)
    integer, intent(in) :: iK, iSpin, nK, nSpin
    integer :: i, j, iKS, n
    if (.not. tStoreMatrices) return
    n = size(DM, dim=1)
    call ensure_cplx_store(n, nK * nSpin)
    iKS = iK + (iSpin - 1) * nK
    do j = 1, n
      do i = j, n
        storedDMCplx(i,j,iKS) = DM(i,j)
        storedDMCplx(j,i,iKS) = conjg(DM(i,j))
      end do
    end do
  end subroutine store_dm_cplx

  !> Store complex eigenvectors (columns are MOs) and eigenvalues for (iK, iSpin).
  subroutine store_eigvecs_cplx(C, evals, iK, iSpin, nK, nSpin)
    complex(dp), intent(in) :: C(:,:)
    real(dp), intent(in) :: evals(:)
    integer, intent(in) :: iK, iSpin, nK, nSpin
    integer :: iKS, n
    if (.not. tStoreMatrices) return
    n = size(C, dim=1)
    call ensure_cplx_store(n, nK * nSpin)
    iKS = iK + (iSpin - 1) * nK
    storedEigvecsCplx(:,:,iKS) = C(1:n, 1:n)
    storedEigvalsCplx(:,iKS)   = evals(1:n)
  end subroutine store_eigvecs_cplx

  !> Query dimensions of the complex store; nKS=0 means no k-point data stored.
  subroutine get_cplx_store_dims(nOrb, nKS)
    integer, intent(out) :: nOrb, nKS
    if (allocated(storedHCplx)) then
      nOrb = nOrbCplx; nKS = nKScplx
    else
      nOrb = 0; nKS = 0
    end if
  end subroutine get_cplx_store_dims

  subroutine get_stored_hamiltonian_cplx(H)
    complex(dp), intent(out) :: H(:,:,:)
    if (allocated(storedHCplx)) then; H = storedHCplx
    else;                             H = cmplx(0.0_dp, 0.0_dp, dp)
    end if
  end subroutine get_stored_hamiltonian_cplx

  subroutine get_stored_overlap_cplx(S)
    complex(dp), intent(out) :: S(:,:,:)
    if (allocated(storedSCplx)) then; S = storedSCplx
    else;                             S = cmplx(0.0_dp, 0.0_dp, dp)
    end if
  end subroutine get_stored_overlap_cplx

  subroutine get_stored_dm_cplx(DM)
    complex(dp), intent(out) :: DM(:,:,:)
    if (allocated(storedDMCplx)) then; DM = storedDMCplx
    else;                              DM = cmplx(0.0_dp, 0.0_dp, dp)
    end if
  end subroutine get_stored_dm_cplx

  subroutine get_stored_eigvecs_cplx(C, evals)
    complex(dp), intent(out) :: C(:,:,:)
    real(dp), intent(out) :: evals(:,:)
    if (allocated(storedEigvecsCplx)) then
      C = storedEigvecsCplx; evals = storedEigvalsCplx
    else
      C = cmplx(0.0_dp, 0.0_dp, dp); evals = 0.0_dp
    end if
  end subroutine get_stored_eigvecs_cplx

  subroutine clear_stored_matrices()
    if (allocated(storedH))        deallocate(storedH)
    if (allocated(storedS))        deallocate(storedS)
    if (allocated(storedDM))       deallocate(storedDM)
    if (allocated(storedEigvecs))  deallocate(storedEigvecs)
    if (allocated(storedEigenvals)) deallocate(storedEigenvals)
    if (allocated(storedHCplx))       deallocate(storedHCplx)
    if (allocated(storedSCplx))       deallocate(storedSCplx)
    if (allocated(storedDMCplx))      deallocate(storedDMCplx)
    if (allocated(storedEigvecsCplx)) deallocate(storedEigvecsCplx)
    if (allocated(storedEigvalsCplx)) deallocate(storedEigvalsCplx)
    if (allocated(cplxSlotFilled))    deallocate(cplxSlotFilled)
    storedSize = 0
    nOrbCplx = 0
    nKScplx = 0
  end subroutine

end module dftbp_dftbplus_hamiltonian_store
