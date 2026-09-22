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
  use dftbp_dftbplus_hsdhelpers, only: doPostParseJobs
  use dftbp_dftbplus_parser, only: parseHsdTree, readHsdFile, TParserFlags
  use dftbp_extlibs_xmlf90, only: destroyNode, fnode
  use dftbp_dftbplus_main, only: runDftbPlus
  use dftbp_io_formatout, only: printDftbHeader
  use dftbp_dftbplus_hamiltonian_store, only: set_store_hamiltonian, get_stored_hamiltonian,&
      & get_stored_overlap, get_stored_dm, get_stored_eigvecs, clear_stored_matrices,&
      & get_stored_hamiltonian_cplx, get_stored_overlap_cplx, get_stored_dm_cplx,&
      & get_stored_eigvecs_cplx, get_cplx_store_dims
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
  logical, save :: tCollectH = .false.
  logical, save :: tCollectS = .false.
  logical, save :: tCollectDM = .false.
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
  public :: dftbcore_get_cplx_dims, dftbcore_get_kpoints
  public :: dftbcore_get_h_cplx, dftbcore_get_s_cplx, dftbcore_get_dm_cplx
  public :: dftbcore_get_eigvecs_cplx

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
    type(fnode), pointer :: hsdTree
    type(TParserFlags) :: parserFlags
    logical :: tExist

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

    ! Parse the HSD input file given by the caller
    inquire(file=trim(hsdPath), exist=tExist)
    if (.not. tExist) then
      print *, '[DFTBcore] ERROR: input file not found: ', trim(hsdPath)
      error stop
    end if
    call readHsdFile(trim(hsdPath), hsdTree)
    call parseHsdTree(hsdTree, input, parserFlags)
    call doPostParseJobs(hsdTree, parserFlags)
    call destroyNode(hsdTree)
    
    ! Initialize environment
    call TEnvironment_init(env)
    
    ! Initialize main program variables
    call main%initProgramVariables(input, env)
    
    ! Deallocate input (no longer needed after init)
    deallocate(input)
    
    ! Always enable storage so store_eigvecs (and optional H/S/DM) can capture data during SCF
    call set_store_hamiltonian(.true.)
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
    
    if (.not. main%tRealHS) then
      print *, '[DFTBcore] Periodic k-point run: H(k)/S(k)/DM(k)/eigvecs stored as complex;'
      print *, '[DFTBcore]   use dftbcore_get_*_cplx getters (dense real getters return zeros).'
    end if

    ! Always extract eigenvectors (stored in hamiltonian_store during SCF via store_eigvecs)
    if (allocated(storedEigvecs))  deallocate(storedEigvecs)
    if (allocated(storedEigenvals)) deallocate(storedEigenvals)
    allocate(storedEigvecs(basisSize, basisSize), source=0.0_dp)
    allocate(storedEigenvals(basisSize), source=0.0_dp)
    call get_stored_eigvecs(storedEigvecs, storedEigenvals, iSpin)
    if (iSpin > 0) then
      print *, '[DFTBcore] Eigenvectors extracted from hamiltonian_store'
    else
      print *, '[DFTBcore] WARNING: Eigenvectors not available in hamiltonian_store'
    end if

    ! Optionally collect H, S, DM
    if (tCollectH .or. tCollectS .or. tCollectDM) then
      if (allocated(storedH))  deallocate(storedH)
      if (allocated(storedS))  deallocate(storedS)
      if (allocated(storedDM)) deallocate(storedDM)
      allocate(storedH(basisSize, basisSize), source=0.0_dp)
      allocate(storedS(basisSize, basisSize), source=0.0_dp)
      allocate(storedDM(basisSize, basisSize), source=0.0_dp)
      
      call get_stored_hamiltonian(storedH, iSpin)
      call get_stored_overlap(storedS, iSpin)
      call get_stored_dm(storedDM, iSpin)
      storedH = 0.5_dp * (storedH + transpose(storedH))
      storedS = 0.5_dp * (storedS + transpose(storedS))
      print *, '[DFTBcore] H/S/DM extracted: basis=', basisSize
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

  ! ---- Complex (k-point) getters for periodic systems ----
  ! Slot index iKS = iK + (iSpin-1)*nKPoint, i.e. k-point index runs fastest.

  !> Query dimensions of the complex (k-point) data store.
  !> nks = 0 means no k-point data was stored (cluster or Gamma-only run).
  subroutine dftbcore_get_cplx_dims(norb, nks, nkpts, nspin) &
      & bind(c, name='dftbcore_get_cplx_dims')
    integer(c_int), intent(out) :: norb, nks, nkpts, nspin
    call get_cplx_store_dims(norb, nks)
    if (allocated(main) .and. allocated(main%kPoint)) then
      nkpts = main%nKPoint
      nspin = main%nIndepSpin
    else
      nkpts = 0
      nspin = 0
    end if
  end subroutine dftbcore_get_cplx_dims

  !> K-points (in fractions of reciprocal lattice vectors) and their weights.
  subroutine dftbcore_get_kpoints(kpts, weights, nk) bind(c, name='dftbcore_get_kpoints')
    integer(c_int), value, intent(in) :: nk
    real(c_double), intent(out) :: kpts(3, nk)
    real(c_double), intent(out) :: weights(nk)
    if (allocated(main) .and. allocated(main%kPoint)) then
      kpts(:, 1:nk) = main%kPoint(:, 1:nk)
      weights(1:nk) = main%kWeight(1:nk)
    else
      kpts = 0.0_dp
      weights = 0.0_dp
    end if
  end subroutine dftbcore_get_kpoints

  subroutine dftbcore_get_h_cplx(hbuf, norb, nks) bind(c, name='dftbcore_get_h_cplx')
    integer(c_int), value, intent(in) :: norb, nks
    complex(c_double_complex), intent(out) :: hbuf(norb, norb, nks)
    call get_stored_hamiltonian_cplx(hbuf)
  end subroutine dftbcore_get_h_cplx

  subroutine dftbcore_get_s_cplx(sbuf, norb, nks) bind(c, name='dftbcore_get_s_cplx')
    integer(c_int), value, intent(in) :: norb, nks
    complex(c_double_complex), intent(out) :: sbuf(norb, norb, nks)
    call get_stored_overlap_cplx(sbuf)
  end subroutine dftbcore_get_s_cplx

  subroutine dftbcore_get_dm_cplx(dbuf, norb, nks) bind(c, name='dftbcore_get_dm_cplx')
    integer(c_int), value, intent(in) :: norb, nks
    complex(c_double_complex), intent(out) :: dbuf(norb, norb, nks)
    call get_stored_dm_cplx(dbuf)
  end subroutine dftbcore_get_dm_cplx

  !> Complex eigenvectors (columns are MOs) and real eigenvalues per (k, spin) slot.
  subroutine dftbcore_get_eigvecs_cplx(cbuf, ebuf, norb, nks) &
      & bind(c, name='dftbcore_get_eigvecs_cplx')
    integer(c_int), value, intent(in) :: norb, nks
    complex(c_double_complex), intent(out) :: cbuf(norb, norb, nks)
    real(c_double), intent(out) :: ebuf(norb, nks)
    call get_stored_eigvecs_cplx(cbuf, ebuf)
  end subroutine dftbcore_get_eigvecs_cplx

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
