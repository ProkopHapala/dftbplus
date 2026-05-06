!--------------------------------------------------------------------------------------------------!
!  DFTBcore: Real C-bindable interface for DFTB+ calculations
!--------------------------------------------------------------------------------------------------!
!
! This module provides a general interface to run actual DFTB+ calculations
! and extract real Hamiltonian, overlap, and density matrices.

#:include 'common.fypp'

module libdftbcore
  use iso_c_binding
  use dftbp_common_accuracy, only: dp
  use dftbp_common_environment, only: TEnvironment, TEnvironment_init
  use dftbp_common_globalenv, only: initGlobalEnv, destructGlobalEnv
  use dftbp_dftbplus_inputdata, only: TInputData
  use dftbp_dftbplus_initprogram, only: TDftbPlusMain
  use dftbp_dftbplus_hsdhelpers, only: parseHsdInput
  use dftbp_dftbplus_main, only: runDftbPlus
  use dftbp_io_formatout, only: printDftbHeader
  use dftbp_dftbplus_hamiltonian_store, only: set_store_hamiltonian, get_stored_hamiltonian,&
      & get_stored_overlap, get_stored_dm, clear_stored_matrices
  implicit none
  private

  ! DFTB+ state - allocated during init, used throughout
  type(TDftbPlusMain), allocatable, target, save :: main
  type(TEnvironment), allocatable, save :: env
  type(TInputData), allocatable, save :: input
  
  ! Matrix storage (allocated after SCF converges)
  real(dp), allocatable, save :: storedH(:,:)      ! Hamiltonian
  real(dp), allocatable, save :: storedS(:,:)      ! Overlap
  real(dp), allocatable, save :: storedDM(:,:)     ! Density matrix
  real(dp), allocatable, save :: storedEigvecs(:,:)  ! Eigenvectors
  real(dp), allocatable, save :: storedEigenvals(:)  ! Eigenvalues
  
  ! Settings
  logical, save :: tCollectH = .true.
  logical, save :: tCollectS = .true.
  logical, save :: tCollectDM = .true.
  logical, save :: tDebug = .false.
  logical, save :: isInitialized = .false.
  integer, save :: basisSize = 0

  public :: dftbcore_init, dftbcore_finalize
  public :: dftbcore_set_coords, dftbcore_set_coords_and_lattice
  public :: dftbcore_enable_matrix_collection, dftbcore_set_debug
  public :: dftbcore_enable_hamiltonian_storage
  public :: dftbcore_run_scf
  public :: dftbcore_write_debug_matrices
  public :: dftbcore_get_basis_size
  public :: dftbcore_get_dm_dense, dftbcore_get_h_dense, dftbcore_get_s_dense
  public :: dftbcore_get_energy
  public :: dftbcore_get_eigvecs_dense

contains

  ! Helper: Convert C string to Fortran string
  subroutine c_to_f_string(c_str, f_str)
    character(c_char), intent(in) :: c_str(*)
    character(len=*), intent(out) :: f_str
    integer :: i
    f_str = ''
    do i = 1, len(f_str)
      if (c_str(i) == c_null_char) exit
      f_str(i:i) = char(ichar(c_str(i)))
    end do
  end subroutine


  subroutine dftbcore_init(inputFile, outputFile) bind(c, name='dftbcore_init')
    character(c_char), intent(in) :: inputFile(*)
    character(c_char), intent(in), optional :: outputFile(*)
    character(256) :: hsdPath
    integer :: iErr
    
    print *, '[DFTBcore] Initializing DFTB+...'
    
    ! Convert C string to Fortran
    call c_to_f_string(inputFile, hsdPath)
    print *, '[DFTBcore] Input file: ', trim(hsdPath)
    
    ! Initialize global environment (MPI, etc.)
    call initGlobalEnv()
    
    ! Allocate state
    allocate(env)
    allocate(input)
    allocate(main)
    
    ! Parse HSD input
    call parseHsdInput(input)
    
    ! Initialize environment
    call TEnvironment_init(env)
    
    ! Initialize main program variables
    call main%initProgramVariables(input, env)
    
    ! Deallocate input (no longer needed after init)
    deallocate(input)
    
    isInitialized = .true.
    print *, '[DFTBcore] Initialization complete'
  end subroutine

  subroutine dftbcore_set_coords(natoms, coords) bind(c, name='dftbcore_set_coords')
    integer(c_int), value, intent(in) :: natoms
    real(c_double), intent(in) :: coords(3, natoms)
    
    if (.not. isInitialized) then
      print *, '[DFTBcore] ERROR: not initialized'
      return
    end if
    
    print *, '[DFTBcore] Setting geometry: ', natoms, ' atoms'
    ! TODO: Update main%coord%coords with new positions
    ! This requires accessing main%coord and calling update routines
  end subroutine

  subroutine dftbcore_set_coords_and_lattice(natoms, coords, lattice) &
      & bind(c, name='dftbcore_set_coords_and_lattice')
    integer(c_int), value, intent(in) :: natoms
    real(c_double), intent(in) :: coords(3, natoms)
    real(c_double), intent(in) :: lattice(3, 3)
    
    if (.not. isInitialized) then
      print *, '[DFTBcore] ERROR: not initialized'
      return
    end if
    
    print *, '[DFTBcore] Setting geometry with lattice: ', natoms, ' atoms'
    ! TODO: Update coords and lattice vectors
  end subroutine

  subroutine dftbcore_enable_matrix_collection(collectDM, collectH, collectS) &
      & bind(c, name='dftbcore_enable_matrix_collection')
    integer(c_int), value, intent(in) :: collectDM, collectH, collectS
    tCollectDM = (collectDM /= 0)
    tCollectH = (collectH /= 0)
    tCollectS = (collectS /= 0)
    print *, '[DFTBcore] Matrix collection: DM=', tCollectDM, ' H=', tCollectH, ' S=', tCollectS
  end subroutine

  subroutine dftbcore_set_debug(debug) bind(c, name='dftbcore_set_debug')
    integer(c_int), value, intent(in) :: debug
    tDebug = (debug /= 0)
    print *, '[DFTBcore] Debug mode:', tDebug
  end subroutine

  subroutine dftbcore_enable_hamiltonian_storage(store) bind(c, name='dftbcore_enable_hamiltonian_storage')
    integer(c_int), value, intent(in) :: store
    call set_store_hamiltonian(store /= 0)
    print *, '[DFTBcore] Hamiltonian storage:', (store /= 0)
  end subroutine

  subroutine dftbcore_run_scf(energy, ierr) bind(c, name='dftbcore_run_scf')
    real(c_double), intent(out) :: energy
    integer(c_int), intent(out) :: ierr
    
    integer :: iSpin
    
    ierr = 0
    energy = 0.0_dp
    
    if (.not. isInitialized) then
      print *, '[DFTBcore] ERROR: not initialized'
      ierr = 1
      return
    end if
    
    print *, '[DFTBcore] Running SCF...'
    
    ! Run the actual DFTB+ calculation
    call runDftbPlus(main, env)
    
    ! Extract energy from dftbEnergy array (index 1 for first determinant)
    if (allocated(main%dftbEnergy)) then
      energy = main%dftbEnergy(1)%Etotal
    else
      energy = 0.0_dp
    end if
    
    ! Get basis size from the main object
    basisSize = main%nOrb
    
    ! Allocate and store matrices if requested
    if (tCollectH .or. tCollectS .or. tCollectDM) then
      if (allocated(storedH)) deallocate(storedH)
      if (allocated(storedS)) deallocate(storedS)
      if (allocated(storedDM)) deallocate(storedDM)
      if (allocated(storedEigvecs)) deallocate(storedEigvecs)
      if (allocated(storedEigenvals)) deallocate(storedEigenvals)
      
      allocate(storedH(basisSize, basisSize), source=0.0_dp)
      allocate(storedS(basisSize, basisSize), source=0.0_dp)
      allocate(storedDM(basisSize, basisSize), source=0.0_dp)
      allocate(storedEigvecs(basisSize, basisSize), source=0.0_dp)
      allocate(storedEigenvals(basisSize), source=0.0_dp)
      
      ! Extract matrices from DFTB+ internal storage
      ! Note: Arrays are column-major in Fortran, Python will transpose to row-major
      ! Try to get Hamiltonian from storage (before diagonalization)
      call get_stored_hamiltonian(storedH, basisSize)
      if (any(storedH /= 0.0_dp)) then
        print *, '[DFTBcore] Using stored Hamiltonian (before diagonalization)'
      else if (allocated(main%HSqrReal)) then
        if (size(main%HSqrReal, 1) >= basisSize .and. size(main%HSqrReal, 2) >= basisSize) then
          storedH = main%HSqrReal(1:basisSize, 1:basisSize)
          print *, '[DFTBcore] Using main%HSqrReal (may contain eigenvectors)'
        end if
      end if
      
      ! Overlap and DM: retrieved from hamiltonian_store (stored at valid points in SCF loop)
      call get_stored_overlap(storedS, iSpin)
      if (iSpin > 0) then
        print *, '[DFTBcore] S from hamiltonian_store'
      else
        print *, '[DFTBcore] WARNING: S not available in hamiltonian_store'
      end if
      call get_stored_dm(storedDM, iSpin)
      if (iSpin > 0) then
        print *, '[DFTBcore] DM from hamiltonian_store'
      else
        print *, '[DFTBcore] WARNING: DM not available in hamiltonian_store'
      end if
      
      ! Extract eigenvectors and eigenvalues
      if (allocated(main%eigVecsReal)) then
        if (size(main%eigVecsReal, 1) >= basisSize .and. size(main%eigVecsReal, 2) >= basisSize) then
          storedEigvecs = main%eigVecsReal(1:basisSize, 1:basisSize, 1)
          print *, '[DFTBcore] Eigenvectors extracted'
        end if
      end if
      
      if (allocated(main%eigen)) then
        if (size(main%eigen, 1) >= basisSize) then
          storedEigenvals = main%eigen(1:basisSize, 1, 1)
          print *, '[DFTBcore] Eigenvalues extracted'
        end if
      end if
      
      ! Symmetrize matrices (unpackHS only fills lower triangle)
      storedH = 0.5_dp * (storedH + transpose(storedH))
      storedS = 0.5_dp * (storedS + transpose(storedS))
      
      print *, '[DFTBcore] Matrices extracted: basis=', basisSize
    end if
    
    print *, '[DFTBcore] SCF complete: E=', energy, 'Hartree'
  end subroutine

  subroutine dftbcore_write_debug_matrices() bind(c, name='dftbcore_write_debug_matrices')
    if (.not. tDebug) then
      return
    end if
    
    if (.not. (allocated(storedH) .and. allocated(storedS) .and. allocated(storedDM))) then
      print *, '[DFTBcore] WARNING: Matrices not allocated, cannot write debug files'
      return
    end if
    
    open(unit=100, file='debug_H.dat', status='replace', action='write')
    write(100, '(6ES24.15)') storedH
    close(100)
    
    open(unit=100, file='debug_S.dat', status='replace', action='write')
    write(100, '(6ES24.15)') storedS
    close(100)
    
    open(unit=100, file='debug_DM.dat', status='replace', action='write')
    write(100, '(6ES24.15)') storedDM
    close(100)
    
    print *, '[DFTBcore] Debug files written: debug_H.dat, debug_S.dat, debug_DM.dat'
  end subroutine

  subroutine dftbcore_get_basis_size(n) bind(c, name='dftbcore_get_basis_size')
    integer(c_int), intent(out) :: n
    n = basisSize
  end subroutine

  subroutine dftbcore_get_energy(energy) bind(c, name='dftbcore_get_energy')
    real(c_double), intent(out) :: energy
    energy = main%dftbEnergy(1)%Etotal
  end subroutine

  subroutine dftbcore_get_eigvecs_dense(eigvecs, eigvals, n) bind(c, name='dftbcore_get_eigvecs_dense')
    integer(c_int), value, intent(in) :: n
    real(c_double), intent(out) :: eigvecs(n*n)
    real(c_double), intent(out) :: eigvals(n)
    if (.not. allocated(storedEigvecs) .or. .not. allocated(storedEigenvals)) then
      print *, '[DFTBcore] WARNING: eigenvectors not available'
      eigvecs = 0.0_dp; eigvals = 0.0_dp; return
    end if
    eigvecs(:) = reshape(storedEigvecs(1:n,1:n), [n*n])
    eigvals(:) = storedEigenvals(1:n)
  end subroutine

  subroutine dftbcore_get_h_dense(h, n) bind(c, name='dftbcore_get_h_dense')
    integer(c_int), value, intent(in) :: n
    real(c_double), intent(out) :: h(n*n)
    if (allocated(storedH)) then
      h(:) = reshape(storedH(1:n,1:n), [n*n])
    else
      h = 0.0_dp; print *, '[DFTBcore] WARNING: H not available'
    end if
  end subroutine

  subroutine dftbcore_get_s_dense(s, n) bind(c, name='dftbcore_get_s_dense')
    integer(c_int), value, intent(in) :: n
    real(c_double), intent(out) :: s(n*n)
    if (allocated(storedS)) then
      s(:) = reshape(storedS(1:n,1:n), [n*n])
    else
      s = 0.0_dp; print *, '[DFTBcore] WARNING: S not available'
    end if
  end subroutine

  subroutine dftbcore_get_dm_dense(dm, n) bind(c, name='dftbcore_get_dm_dense')
    integer(c_int), value, intent(in) :: n
    real(c_double), intent(out) :: dm(n*n)
    if (allocated(storedDM)) then
      dm(:) = reshape(storedDM(1:n,1:n), [n*n])
    else
      dm = 0.0_dp; print *, '[DFTBcore] WARNING: DM not available'
    end if
  end subroutine

  subroutine dftbcore_finalize() bind(c, name='dftbcore_finalize')
    print *, '[DFTBcore] Finalizing...'
    
    if (allocated(storedH)) deallocate(storedH)
    if (allocated(storedS)) deallocate(storedS)
    if (allocated(storedDM)) deallocate(storedDM)
    if (allocated(storedEigvecs)) deallocate(storedEigvecs)
    if (allocated(storedEigenvals)) deallocate(storedEigenvals)
    
    ! Clear stored Hamiltonian
    call clear_stored_matrices()
    
    if (allocated(main)) then
      call main%destructProgramVariables()
      deallocate(main)
    end if
    
    if (allocated(env)) then
      call env%destruct()
      deallocate(env)
    end if
    
    call destructGlobalEnv()
    isInitialized = .false.
    basisSize = 0
    
    print *, '[DFTBcore] Finalized'
  end subroutine

end module libdftbcore
