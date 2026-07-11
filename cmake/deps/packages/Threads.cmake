deps_declare(Threads
    REQUIRED             TRUE
    APT                  ON
    APT_METHOD           FIND_PACKAGE
    APT_FIND_NAME        Threads
    APT_IMPORTED_TARGETS "Threads::Threads"
)
