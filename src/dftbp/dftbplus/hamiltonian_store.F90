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

  real(dp), allocatable, save :: storedH(:,:)
  real(dp), allocatable, save :: storedS(:,:)
  real(dp), allocatable, save :: storedDM(:,:)
  real(dp), allocatable, save :: storedEigvecs(:,:)   ! (norb, nstates) for iKS=1, iSpin=1
  real(dp), allocatable, save :: storedEigenvals(:)   ! (nstates) for iKS=1, iSpin=1
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

  subroutine clear_stored_matrices()
    if (allocated(storedH))        deallocate(storedH)
    if (allocated(storedS))        deallocate(storedS)
    if (allocated(storedDM))       deallocate(storedDM)
    if (allocated(storedEigvecs))  deallocate(storedEigvecs)
    if (allocated(storedEigenvals)) deallocate(storedEigenvals)
    storedSize = 0
  end subroutine

end module dftbp_dftbplus_hamiltonian_store
