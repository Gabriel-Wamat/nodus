# Security policy

## Reporting

Report suspected vulnerabilities privately through GitHub Security Advisories for this
repository. Do not open a public issue containing credentials, private cluster endpoints or
exploit details.

## Supported version

The latest version on the default branch receives security fixes during the pre-1.0 phase.

## Security model

Nodus delegates host verification and authentication to OpenSSH, does not persist passwords or
private keys, confines remote state to the configured user-owned root, and only cancels jobs
recorded in its local state store. It does not require administrative privileges or modify
scheduler configuration.
