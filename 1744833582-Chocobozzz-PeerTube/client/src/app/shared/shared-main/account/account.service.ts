import { Observable, ReplaySubject } from 'rxjs'
import { catchError, map, tap } from 'rxjs/operators'
import { HttpClient } from '@angular/common/http'
import { Injectable, inject } from '@angular/core'
import { RestExtractor } from '@app/core'
import { Account as ServerAccount } from '@peertube/peertube-models'
import { environment } from '../../../../environments/environment'
import { Account } from './account.model'

@Injectable()
export class AccountService {
  private authHttp = inject(HttpClient)
  private restExtractor = inject(RestExtractor)

  static BASE_ACCOUNT_URL = environment.apiUrl + '/api/v1/accounts/'

  accountLoaded = new ReplaySubject<Account>(1)

  getAccount (id: number | string): Observable<Account> {
    return this.authHttp.get<ServerAccount>(AccountService.BASE_ACCOUNT_URL + id)
      .pipe(
        map(accountHash => new Account(accountHash)),
        tap(account => this.accountLoaded.next(account)),
        catchError(res => this.restExtractor.handleError(res))
      )
  }
}
